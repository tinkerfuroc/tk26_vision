#!/usr/bin/env python3
"""Export Fast-FoundationStereo two-stage ONNX and build FP16 TensorRT engines.

    python -m foundation_stereo.scripts.build_trt_engines [--weights-root DIR] [--force]

Steps: (1) run the vendored scripts/make_onnx.py on the 23-36-37 pickle at
576x960 / 4 iters / max_disp 192 into a temp dir; (2) build feature_runner
and post_runner engines with the TensorRT Python API (trtexec is not shipped
in the pip wheels); (3) install {feature_runner.engine, post_runner.engine,
onnx.yaml} as <weights_root>/Fast-FoundationStereo/<variant>/ — the layout
stereo_runner._discover_trt_variants expects. Engines are GPU/TRT-locked:
rebuild on every new box.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from foundation_stereo.scripts.fetch_fast_fs_weights import DEFAULT_WEIGHTS_ROOT, checkpoint_path

ENGINE_FILES = ("feature_runner.engine", "post_runner.engine", "onnx.yaml")


def variant_dir(weights_root: str, name: str = "output_two_stage") -> Path:
    return Path(os.path.expanduser(weights_root)) / "Fast-FoundationStereo" / name


def vendor_fast_dir() -> Path:
    """Locate thirdparty/foundation_stereo/Fast-FoundationStereo like stereo_runner does."""
    env = os.environ.get("FOUNDATION_STEREO_VENDOR_ROOT")
    if env:
        return Path(os.path.expanduser(env)) / "Fast-FoundationStereo"
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "thirdparty" / "foundation_stereo" / "Fast-FoundationStereo"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("Fast-FoundationStereo vendor tree not found; set FOUNDATION_STEREO_VENDOR_ROOT")


def make_onnx_command(fast_dir: Path, ckpt: Path, save_dir: Path,
                      height: int, width: int, valid_iters: int, max_disp: int) -> list[str]:
    return [sys.executable, str(fast_dir / "scripts" / "make_onnx.py"),
            "--model_dir", str(ckpt), "--save_path", str(save_dir),
            "--height", str(height), "--width", str(width),
            "--valid_iters", str(valid_iters), "--max_disp", str(max_disp)]


def ensure_writable(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any((out_dir / n).exists() for n in ENGINE_FILES) and not force:
        sys.exit(f"{out_dir} already holds engines; pass --force to rebuild")


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True, workspace_gib: int = 4) -> float:
    import tensorrt as trt  # noqa: WPS433 — only available in .venv-fs
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fp:
        if not parser.parse(fp.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("warning: platform reports no fast fp16; building fp16 anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as fp:
        fp.write(blob)
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights-root", default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument("--variant", default="output_two_stage")
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--valid-iters", type=int, default=4)
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--workspace-gib", type=int, default=4)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ckpt = checkpoint_path(args.weights_root)
    if not ckpt.is_file():
        sys.exit(f"checkpoint missing: {ckpt} — run fetch_fast_fs_weights first")
    out_dir = variant_dir(args.weights_root, args.variant)
    ensure_writable(out_dir, args.force)
    fast_dir = vendor_fast_dir()

    with tempfile.TemporaryDirectory(prefix="fs_onnx_") as tmp:
        tmp = Path(tmp)
        cmd = make_onnx_command(fast_dir, ckpt, tmp, args.height, args.width, args.valid_iters, args.max_disp)
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(fast_dir))   # make_onnx imports `core` relative to its tree
        for stem in ("feature_runner", "post_runner"):
            secs = build_engine(tmp / f"{stem}.onnx", tmp / f"{stem}.engine",
                                fp16=not args.no_fp16, workspace_gib=args.workspace_gib)
            print(f"built {stem}.engine in {secs:.0f} s")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ENGINE_FILES:
            shutil.copy2(tmp / name, out_dir / name)
    print(f"installed {out_dir}: {sorted(p.name for p in out_dir.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
