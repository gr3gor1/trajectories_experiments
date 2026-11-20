"""
Measured (not projected) speedup of every routing criterion on the real video.

    python -m adaptive_yolo.benchmark results/ground_level_0148-0351 --model 8n --weights 8n.pt \
        --source GROUND_LEVEL.MP4 --own-vessel-bottom 0.97

For every frame of the evaluated window (the frames in <model>_cache.pkl):

  1. the video frame is decoded once                                  → decode_ms
  2. on that same decoded frame, in a shuffled order, GPU-synchronised:
        model(frame, P3)                                              → P3_ms
        model(frame, P3_P4)                                           → P3_P4_ms
        model(frame, FULL)                                            → FULL_ms
        model.detect_all_levels(frame)  (FULL pass + all decodes)     → FULL_learn_ms
     each timing covers what a deployed router pays per frame: pre-processing (resize,
     colour conversion, upload), forward pass, box decoding and NMS.

Every criterion is then charged, frame by frame, the measured cost of the level it chose in
the evaluation (<model>_frames.csv), so all criteria are compared on identical frames and
identical GPU conditions. Learned routers pay FULL_learn_ms on their FULL frames (that is the
pass they learn from). Routing overhead (the router's own decision/update time, CPU) is measured
by replaying each criterion and added; the oracle needs hindsight and has none.

    model-only speedup  = Σ FULL_ms / Σ criterion_ms
    end-to-end speedup  = Σ (decode_ms + FULL_ms) / Σ (decode_ms + criterion_ms)

Outputs: <dir>/benchmark/<model>_benchmark_frames.csv (per-frame timings) and
<dir>/benchmark/<model>_benchmark.csv (per criterion), merged into the aggregate tables by
`python -m adaptive_yolo.aggregate`.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .evaluate_routers import Oracle, policies
from .learned import LearnedRouter, drop_own_vessel
from .model import AdaptiveYOLO, ExitLevel

VARIANTS = ("P3", "P3_P4", "FULL", "FULL_learn")


def time_levels(model: AdaptiveYOLO, cap, start_frame: int, n_frames: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {start_frame}")
    for _ in range(10):  # warm up every path on a real frame
        for lvl in ExitLevel:
            model(first, lvl)
        model.detect_all_levels(first)
    model.sync()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows = []
    for k in range(n_frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        decode = (time.perf_counter() - t0) * 1000
        if not ok:
            break
        row = {"frame": start_frame + k, "decode_ms": decode}
        for v in rng.permutation(VARIANTS):
            model.sync()
            t0 = time.perf_counter()
            if v == "FULL_learn":
                model.detect_all_levels(frame)
            else:
                model(frame, ExitLevel[v])
            model.sync()
            row[f"{v}_ms"] = (time.perf_counter() - t0) * 1000
        rows.append(row)
        if (k + 1) % 500 == 0:
            print(f"  frame {k + 1} / {n_frames}")
    return pd.DataFrame(rows)


def routing_overhead(cache: dict, own_vessel_bottom, label_conf: float, targets, probe: float, seed: int) -> dict:
    """Mean CPU time (ms/frame) of each criterion's own decision (+ learning) step."""
    frames = cache["frames"]
    if own_vessel_bottom is not None:
        frames = [(r, drop_own_vessel(d, cache["frame_h"], own_vessel_bottom)) for r, d in frames]
    out = {}
    for name, router in policies(targets, probe, label_conf, seed).items():
        if isinstance(router, Oracle):
            out[name] = 0.0  # hindsight only, not deployable
            continue
        if hasattr(router, "reset"):
            router.reset()
        t0 = time.perf_counter()
        for reading, dets in frames:
            lvl = router(reading)
            if isinstance(router, LearnedRouter) and lvl == ExitLevel.FULL:
                router.observe(dets)
        out[name] = (time.perf_counter() - t0) * 1000 / len(frames)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--model", required=True, help="model tag used in the results files, e.g. 8n")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--own-vessel-bottom", type=float, default=None)
    p.add_argument("--label-conf", type=float, default=0.25)
    p.add_argument("--targets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--probe", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    res = Path(a.results)
    out_dir = res / "benchmark"
    out_dir.mkdir(exist_ok=True)
    cache = pickle.loads((res / f"{a.model}_cache.pkl").read_bytes())
    decisions = pd.read_csv(res / f"{a.model}_frames.csv")
    model = AdaptiveYOLO(a.weights, device=a.device, conf=a.label_conf)

    cap = cv2.VideoCapture(a.source)
    t = time_levels(model, cap, cache.get("start_frame", 0), len(cache["frames"]), a.seed)
    cap.release()
    t.to_csv(out_dir / f"{a.model}_benchmark_frames.csv", index=False)
    print(f"[benchmark] {len(t)} frames timed; median ms: " +
          ", ".join(f"{v} {t[f'{v}_ms'].median():.2f}" for v in ("decode",) + VARIANTS))

    overhead = routing_overhead(cache, a.own_vessel_bottom, a.label_conf, a.targets, a.probe, a.seed)
    d = decisions.iloc[:len(t)].reset_index(drop=True)
    full_model = t["FULL_ms"].sum()
    full_e2e = (t["decode_ms"] + t["FULL_ms"]).sum()
    proj = pd.read_csv(res / f"{a.model}_summary.csv").set_index("policy")["speedup"]

    rows = []
    for name in [c for c in d.columns if c in proj.index]:
        lv = d[name]
        learned = name.startswith("learned_")
        cost = np.select([lv == "P3", lv == "P3_P4", lv == "FULL"],
                         [t["P3_ms"], t["P3_P4_ms"], t["FULL_learn_ms"] if learned else t["FULL_ms"]])
        cost = cost + overhead.get(name, 0.0)
        rows.append({
            "policy": name,
            "projected_speedup": proj[name],
            "measured_model_ms_per_frame": cost.mean(),
            "measured_e2e_ms_per_frame": (t["decode_ms"] + cost).mean(),
            "measured_model_speedup": full_model / cost.sum(),
            "measured_e2e_speedup": full_e2e / (t["decode_ms"] + cost).sum(),
            "routing_overhead_ms": overhead.get(name, 0.0),
        })
    s = pd.DataFrame(rows)
    s.to_csv(out_dir / f"{a.model}_benchmark.csv", index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
