"""Benchmark: best batch_size + Gemini vs Qwen (quality + performance).

Runs a grid of (photo x provider x batch_size x repeat) single-provider scans
and scores against a hand-labelled ground truth. On the operator's table
photos the ONLY genuine vocab match is a shirt (people wearing shirts / the
crumpled garment); every food/drink/dish/fruit claim is a false positive. So
this doubles as a precision / false-positive benchmark (what the operator asked
to minimise).

    python batch_provider_bench.py            # runs the grid, prints tables
Results JSON -> bench_results.json
"""

from __future__ import annotations

import json
import os
import time

import scan_core

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(HERE, "photos")

# --- ground truth ---------------------------------------------------------- #
SHIRT_CLASSES = {"white shirt", "grey shirt", "blue shirt", "black shirt", "shirt"}

# Table scenes: Korean water bottles + cheese-cracker boxes + bread bag + a
# purple garment + people in shirts. No RoboCup food/drink/dish/fruit actually
# present -> the correct answer is a shirt-class and nothing else.
TABLE_PHOTOS = [
    "photo_20260704_003940_594_ros.jpg",
    "photo_20260704_003943_552_ros.jpg",
    "photo_20260704_004003_506_ros.jpg",
]

PROVIDERS = ["gemini", "qwen"]
BATCH_SIZES = [4, 8, 16, 32]
REPEATS = 2


def score(found):
    found = list(found)
    fp = [l for l in found if l not in SHIRT_CLASSES]     # hallucinated objects
    shirt_hit = any(l in SHIRT_CLASSES for l in found)
    return fp, shirt_hit


def main():
    scan_core.load_env()
    vocab = scan_core.parse_vocabulary()
    photos = [p for p in TABLE_PHOTOS if os.path.isfile(os.path.join(PHOTOS_DIR, p))]
    print(f"vocab={len(vocab)} classes | photos={len(photos)} | providers={PROVIDERS} "
          f"| batch_sizes={BATCH_SIZES} | repeats={REPEATS}")
    urls = {p: scan_core.path_to_data_url(os.path.join(PHOTOS_DIR, p)) for p in photos}

    runs = []
    total = len(photos) * len(PROVIDERS) * len(BATCH_SIZES) * REPEATS
    i = 0
    for photo in photos:
        for prov in PROVIDERS:
            for bs in BATCH_SIZES:
                for rep in range(REPEATS):
                    i += 1
                    res = scan_core.scan_image(
                        urls[photo], vocab, batch_size=bs, providers=(prov,),
                        timeout_s=20.0, max_retries=2)
                    d = res.to_dict()
                    fp, shirt_hit = score(d["found_labels"])
                    runs.append({
                        "photo": photo, "provider": prov, "batch_size": bs,
                        "repeat": rep, "found": d["found_labels"],
                        "n_found": d["n_found"], "n_fp": len(fp), "fp": fp,
                        "shirt_hit": shirt_hit, "latency_s": d["total_latency_s"],
                        "batches_fail": d["batches_fail"],
                    })
                    print(f"[{i}/{total}] {photo[-18:]} {prov:>6} bs={bs:>2} "
                          f"r{rep}: found={d['n_found']} fp={len(fp)} "
                          f"shirt={'Y' if shirt_hit else '-'} {d['total_latency_s']}s "
                          f"{fp if fp else ''}", flush=True)

    with open(os.path.join(HERE, "bench_results.json"), "w") as f:
        json.dump({"runs": runs, "shirt_classes": sorted(SHIRT_CLASSES),
                   "batch_sizes": BATCH_SIZES, "providers": PROVIDERS,
                   "repeats": REPEATS, "photos": photos}, f, indent=2)

    # ---- aggregate ----
    def agg(filt):
        rs = [r for r in runs if filt(r)]
        if not rs:
            return None
        n = len(rs)
        return {
            "n": n,
            "avg_fp": round(sum(r["n_fp"] for r in rs) / n, 2),
            "max_fp": max(r["n_fp"] for r in rs),
            "shirt_rate": round(sum(r["shirt_hit"] for r in rs) / n, 2),
            "avg_latency": round(sum(r["latency_s"] for r in rs) / n, 2),
            "avg_found": round(sum(r["n_found"] for r in rs) / n, 2),
        }

    print("\n================ BY BATCH SIZE (both providers) ================")
    print(f"{'batch':>6} {'avg_fp':>7} {'max_fp':>7} {'shirt':>6} {'lat_s':>6} {'avg_found':>10}")
    for bs in BATCH_SIZES:
        a = agg(lambda r: r["batch_size"] == bs)
        print(f"{bs:>6} {a['avg_fp']:>7} {a['max_fp']:>7} {a['shirt_rate']:>6} "
              f"{a['avg_latency']:>6} {a['avg_found']:>10}")

    print("\n================ BY PROVIDER ================")
    print(f"{'prov':>7} {'avg_fp':>7} {'max_fp':>7} {'shirt':>6} {'lat_s':>6} {'avg_found':>10}")
    for prov in PROVIDERS:
        a = agg(lambda r: r["provider"] == prov)
        print(f"{prov:>7} {a['avg_fp']:>7} {a['max_fp']:>7} {a['shirt_rate']:>6} "
              f"{a['avg_latency']:>6} {a['avg_found']:>10}")

    print("\n================ PROVIDER x BATCH SIZE (avg_fp / lat_s) ================")
    print(f"{'prov':>7} " + " ".join(f"bs{bs:>2}" for bs in BATCH_SIZES))
    for prov in PROVIDERS:
        cells = []
        for bs in BATCH_SIZES:
            a = agg(lambda r: r["provider"] == prov and r["batch_size"] == bs)
            cells.append(f"{a['avg_fp']:.1f}/{a['avg_latency']:.1f}")
        print(f"{prov:>7} " + " ".join(f"{c:>6}" for c in cells))

    # most-hallucinated labels
    from collections import Counter
    for prov in PROVIDERS:
        c = Counter(l for r in runs if r["provider"] == prov for l in r["fp"])
        print(f"\ntop false positives [{prov}]: {c.most_common(8)}")


if __name__ == "__main__":
    main()
