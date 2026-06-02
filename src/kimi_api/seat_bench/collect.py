"""Build seat_bench/dataset/ from logged vision_log seat images.

Scans for *seat*orig*.jpg, dedupes byte-identical files (the 'copy'
sessions are literal cp -r duplicates), copies each distinct scene to
dataset/<id>.jpg, and pairs the matching *_req_*.json so strategies can
replay realistic names/features/known_seats. Writes dataset/manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .paths import DATASET_DIR, find_vision_log


def find_seat_origs() -> list[Path]:
    root = find_vision_log()
    return sorted(root.rglob("*seat*orig*.jpg"))


def dedupe_by_content(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    distinct: list[Path] = []
    for p in paths:
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            distinct.append(p)
    return distinct


def req_path_for_orig(orig: Path) -> Path:
    name = orig.name.replace("_orig_", "_req_")
    name = name.rsplit(".", 1)[0] + ".json"
    return orig.with_name(name)


def _load_req(orig: Path) -> dict:
    req = req_path_for_orig(orig)
    if not req.is_file():
        return {"names": [], "features": [], "known_seats": []}
    data = json.loads(req.read_text()).get("request", {})
    return {
        "names": data.get("names", []),
        "features": data.get("features", []),
        "known_seats": data.get("known_seats", []),
    }


def build() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    origs = find_seat_origs()
    distinct = dedupe_by_content(origs)
    manifest = []
    for i, src in enumerate(distinct):
        sid = f"scene_{i:03d}"
        dst_img = DATASET_DIR / f"{sid}.jpg"
        shutil.copyfile(src, dst_img)
        req = _load_req(src)
        (DATASET_DIR / f"{sid}.req.json").write_text(json.dumps(req, indent=2))
        manifest.append({"id": sid, "src": str(src), **req})
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"collected {len(distinct)} distinct scenes (from {len(origs)} origs)"
          f" -> {DATASET_DIR}")
    return DATASET_DIR


if __name__ == "__main__":
    build()
