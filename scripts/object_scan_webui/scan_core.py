"""Batched labels-only VLM scan — reference implementation.

Standalone (no ROS): splits a candidate vocabulary into batches, asks a vision
LLM per batch which candidates are visible in ONE photo, validates the answer
against the vocabulary (drops hallucinations), and unions the results. Gemini
(OpenRouter) primary, Qwen (DashScope) fallback — same model selection as
`object_detection_generalist`. This module is the prototype that later lifts
into `kimi_api/_scan_vlm.py`; keep it importable without rclpy.

Run as a CLI for a quick check:
    python scan_core.py path/to/photo.jpg --batch-size 8
"""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import openai

# --- model selection (mirrors object_detection_generalist) -----------------
GEMINI_MODEL = "google/gemini-2.5-flash"   # primary, via OpenRouter
QWEN_MODEL = "qwen3-vl-plus"               # fallback, via DashScope
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

_SYS_PROMPT = (
    "You are a precise visual object detector. You are given ONE photo of a "
    "scene and a list of candidate object names. Return a JSON array "
    "containing exactly the candidate names -- copied verbatim from the list "
    "-- that are clearly visible in the photo. Only include a name if you are "
    "confident the object is present. Never include a name that is not in the "
    "list. If none are present, return []. Output ONLY the JSON array, nothing "
    "else."
)


# --------------------------------------------------------------------------- #
# env + workspace helpers                                                      #
# --------------------------------------------------------------------------- #
def find_ws_root(start: Optional[str] = None) -> str:
    """Walk up from `start` (default: this file) to the dir holding src/."""
    here = os.path.abspath(start or __file__)
    for _ in range(12):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "src", "tk25_decision")):
            return here
    return os.getcwd()


def load_env() -> None:
    """Best-effort load of the workspace-root .env (python-dotenv optional)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = find_ws_root()
    for cand in (os.path.join(root, ".env"), ".env"):
        if os.path.isfile(cand):
            load_dotenv(cand)
            return
    load_dotenv()


def parse_vocabulary(constants_path: Optional[str] = None) -> list[str]:
    """Read the PickAndPlace `table_scan_prompt` and split into class names."""
    if constants_path is None:
        constants_path = os.path.join(
            find_ws_root(), "src", "tk25_decision", "src", "behavior_tree",
            "behavior_tree", "PickAndPlace", "constants.json",
        )
    with open(constants_path, "r") as f:
        data = json.load(f)
    prompt = str(data["table_scan_prompt"])
    return [c.strip() for c in prompt.split(" . ") if c.strip()]


def dashscope_api_key() -> Optional[str]:
    # Accept the correct spelling and the legacy typo the .env carries.
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASHCOPE_API_KEY")


def openrouter_api_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY")


# --------------------------------------------------------------------------- #
# image helpers                                                                #
# --------------------------------------------------------------------------- #
def bytes_to_data_url(img_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def path_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(path, "rb") as f:
        return bytes_to_data_url(f.read(), mime)


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip() if "```" in text else text


# --------------------------------------------------------------------------- #
# core scan                                                                    #
# --------------------------------------------------------------------------- #
def batches(vocab: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        return [list(vocab)]
    return [vocab[i:i + batch_size] for i in range(0, len(vocab), batch_size)]


def _client_for(provider: str):
    if provider == "gemini":
        key = openrouter_api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return openai.OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL), GEMINI_MODEL
    if provider == "qwen":
        key = dashscope_api_key()
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY / DASHCOPE_API_KEY not set")
        return openai.OpenAI(api_key=key, base_url=DASHSCOPE_BASE_URL), QWEN_MODEL
    raise RuntimeError(f"unknown provider {provider!r}")


def _validate_labels(raw_text: str, candidates: list[str]) -> list[str]:
    """Parse a JSON array and keep only entries matching a candidate.

    Case-insensitive match; returns the candidate's original casing. Drops
    anything not in the vocabulary (hallucinations). Raises ValueError if the
    payload is not a list (structurally unparseable -> retry/fallback).
    """
    parsed = None
    cleaned = strip_fences(raw_text or "")
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(cleaned)
            break
        except Exception:
            continue
    if not isinstance(parsed, list):
        raise ValueError(f"not a JSON list: {raw_text!r}")
    lut = {c.lower(): c for c in candidates}
    out, seen = [], set()
    for item in parsed:
        key = str(item).strip().lower()
        if key in lut and lut[key] not in seen:
            out.append(lut[key])
            seen.add(lut[key])
    return out


def scan_batch(
    image_data_url: str,
    candidates: list[str],
    *,
    provider_chain=("gemini", "qwen"),
    timeout_s: float = 20.0,
    max_retries: int = 2,
    log=None,
) -> dict:
    """Scan ONE batch. Gemini primary -> Qwen fallback (errors-only).

    An empty list is a legitimate terminal answer (nothing here) and does NOT
    trigger fallback. Only API errors or unparseable responses fall through.
    Returns a dict describing the batch outcome.
    """
    user_content = [
        {"type": "text",
         "text": f"Candidate object names: {json.dumps(candidates)}. "
                 "Which of these are visible in the photo?"},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    messages = [
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]

    errors = []
    for provider in provider_chain:
        try:
            client, model = _client_for(provider)
        except RuntimeError as exc:
            errors.append(f"{provider}: {exc}")
            continue
        t0 = time.perf_counter()
        try:
            for attempt in range(1, max_retries + 1):
                try:
                    comp = client.with_options(timeout=timeout_s)\
                        .chat.completions.create(model=model, messages=messages)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{provider} api(attempt {attempt}): {exc}")
                    if log:
                        log(f"[{provider}] api failed "
                            f"({attempt}/{max_retries}): {exc}")
                    if attempt < max_retries:
                        time.sleep(0.5 * attempt)
                    continue
                raw = comp.choices[0].message.content
                try:
                    labels = _validate_labels(raw, candidates)
                except ValueError as exc:
                    errors.append(f"{provider} parse(attempt {attempt}): {exc}")
                    if log:
                        log(f"[{provider}] parse failed "
                            f"({attempt}/{max_retries}): {exc}")
                    continue
                return {
                    "items": candidates,
                    "found": labels,
                    "provider": provider,
                    "model": model,
                    "latency_s": round(time.perf_counter() - t0, 3),
                    "raw": raw,
                    "error": None,
                }
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        errors.append(f"{provider}: exhausted {max_retries} attempts")
        if log:
            log(f"[{provider}] exhausted; trying next provider")

    return {
        "items": candidates,
        "found": [],
        "provider": None,
        "model": None,
        "latency_s": None,
        "raw": None,
        "error": " | ".join(errors) or "all providers unavailable",
    }


@dataclass
class ScanResult:
    found_labels: list = field(default_factory=list)
    batches: list = field(default_factory=list)
    total_latency_s: float = 0.0
    batches_ok: int = 0
    batches_fail: int = 0
    batch_size: int = 0
    n_vocab: int = 0

    def to_dict(self) -> dict:
        return {
            "found_labels": self.found_labels,
            "batches": self.batches,
            "total_latency_s": round(self.total_latency_s, 3),
            "batches_ok": self.batches_ok,
            "batches_fail": self.batches_fail,
            "batch_size": self.batch_size,
            "n_vocab": self.n_vocab,
            "n_found": len(self.found_labels),
        }


def scan_image(
    image_data_url: str,
    vocabulary: list[str],
    *,
    batch_size: int = 8,
    max_workers: int = 0,
    use_qwen_fallback: bool = True,
    timeout_s: float = 20.0,
    max_retries: int = 2,
    log=None,
) -> ScanResult:
    """Split the vocabulary, scan each batch concurrently, union the results.

    ALL batches fire in parallel by default (`max_workers=0`): one thread per
    batch, so total latency is ~one VLM call regardless of batch count. Set a
    positive `max_workers` only to cap concurrency (e.g. against provider rate
    limits).
    """
    chain = ("gemini", "qwen") if use_qwen_fallback else ("gemini",)
    groups = batches(vocabulary, batch_size)
    # 0 / negative -> one worker per batch (every batch call in parallel).
    workers = len(groups) if max_workers <= 0 else min(max_workers, len(groups))
    workers = max(1, workers)
    t0 = time.perf_counter()

    def _run(batch):
        return scan_batch(
            image_data_url, batch, provider_chain=chain,
            timeout_s=timeout_s, max_retries=max_retries, log=log,
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run, groups))

    total = time.perf_counter() - t0
    # Union preserving vocabulary order.
    found_set = set()
    for r in results:
        found_set.update(r["found"])
    found_labels = [c for c in vocabulary if c in found_set]
    ok = sum(1 for r in results if r["error"] is None)

    return ScanResult(
        found_labels=found_labels,
        batches=results,
        total_latency_s=total,
        batches_ok=ok,
        batches_fail=len(results) - ok,
        batch_size=batch_size,
        n_vocab=len(vocabulary),
    )


def sweep_batch_sizes(
    image_data_url: str,
    vocabulary: list[str],
    batch_sizes: list[int],
    *,
    max_workers: int = 0,
    use_qwen_fallback: bool = True,
    timeout_s: float = 20.0,
    max_retries: int = 2,
    log=None,
) -> list[dict]:
    """Run scan_image for each batch_size; return comparable summaries."""
    out = []
    for bs in batch_sizes:
        res = scan_image(
            image_data_url, vocabulary, batch_size=bs, max_workers=max_workers,
            use_qwen_fallback=use_qwen_fallback, timeout_s=timeout_s,
            max_retries=max_retries, log=log,
        )
        out.append(res.to_dict())
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    load_env()
    ap = argparse.ArgumentParser(description="Batched labels-only VLM scan")
    ap.add_argument("image", help="path to a photo")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-workers", type=int, default=0,
                    help="0 = one worker per batch (all in parallel); >0 caps it")
    ap.add_argument("--no-qwen", action="store_true", help="disable Qwen fallback")
    ap.add_argument("--sweep", default="", help="comma list of batch sizes, e.g. 4,8,16")
    args = ap.parse_args()

    vocab = parse_vocabulary()
    url = path_to_data_url(args.image)
    print(f"vocabulary: {len(vocab)} classes")

    if args.sweep:
        sizes = [int(x) for x in args.sweep.split(",") if x.strip()]
        rows = sweep_batch_sizes(
            url, vocab, sizes, max_workers=args.max_workers,
            use_qwen_fallback=not args.no_qwen, log=print,
        )
        print("\nbatch_size  n_found  latency_s  labels")
        for r in rows:
            print(f"{r['batch_size']:>10}  {r['n_found']:>7}  "
                  f"{r['total_latency_s']:>9}  {r['found_labels']}")
        return

    res = scan_image(
        url, vocab, batch_size=args.batch_size, max_workers=args.max_workers,
        use_qwen_fallback=not args.no_qwen, log=print,
    )
    d = res.to_dict()
    print(f"\nfound {d['n_found']}/{d['n_vocab']} in {d['total_latency_s']}s "
          f"(batches ok={d['batches_ok']} fail={d['batches_fail']})")
    print("labels:", d["found_labels"])
    for i, b in enumerate(d["batches"]):
        print(f"  batch {i}: found {b['found']} via {b['provider']} "
              f"({b['latency_s']}s)" + (f" ERR {b['error']}" if b["error"] else ""))


if __name__ == "__main__":
    main()
