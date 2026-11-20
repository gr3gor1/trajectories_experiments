"""
Compare routing policies (heuristic vs learned) on one video + trajectory log.

    python -m adaptive_yolo.evaluate_routers --weights models/nano_lr.pt \
        --source clip.mp4 --csv log.csv

Pass 1 (GPU, once): for every frame, one FULL forward decoded as all three
levels (exact — see AdaptiveYOLO.detect_all_levels) + the causal trajectory
reading. Cached to <out>_cache.pkl; re-runs with --cache skip this pass.

Pass 2 (CPU, fast): replay each policy over the cached frames. A learned policy
only receives labels on frames where it chose FULL, exactly as it would live.

Metrics, all relative to always-FULL:
    speedup        projected from profiled per-level forward latency
    recall         share of FULL detections (conf >= label_conf) also found by the chosen level
    frame_miss     share of frames with >= 1 FULL detection where the chosen level missed one
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import pandas as pd

from .learned import LearnedRouter, count_missed, drop_own_vessel
from .model import AdaptiveYOLO, ExitLevel
from .router import GradedRouter
from .run_video import frame_range
from .trajectory import TrajectoryLog


# ── pass 1 ────────────────────────────────────────────────────────────────────

def collect(a) -> dict:
    model = AdaptiveYOLO(a.weights, device=a.device, imgsz=a.imgsz, conf=a.conf, iou=a.iou)
    log = TrajectoryLog.from_csv(a.csv, *(a.mmsi or (None, None)))
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_start = log.video_start_from_sync(a.sync) if a.sync else a.video_start

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f, end_f = frame_range(a.start, a.end, fps, n_total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frames, f = [], start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((log.for_frame(f, fps, video_start, a.offset), model.detect_all_levels(frame)))
        f += 1
        if (f - start_f) % 300 == 0:
            print(f"  frame {f} / {end_f}")
    cap.release()
    n_read = sum(r is not None for r, _ in frames)
    print(f"[collect] frames {start_f}–{f - 1}: {len(frames)} frames, {n_read} with trajectory data; profiling …")
    prof = model.profile(runs=a.profile_runs)
    return {"frames": frames, "latency_ms": {l.name: v for l, v in prof.items()}, "fps": fps,
            "start_frame": start_f, "frame_h": frame_h}


# ── pass 2 ────────────────────────────────────────────────────────────────────

class Oracle:
    """Cheapest level that misses nothing on this frame (upper bound, needs hindsight)."""
    def __init__(self, label_conf, iou):
        self.label_conf, self.iou = label_conf, iou
        self.frame = None

    def __call__(self, reading):
        for lvl in (ExitLevel.P3, ExitLevel.P3_P4):
            if count_missed(self.frame[ExitLevel.FULL], self.frame[lvl], self.iou, self.label_conf)[1] == 0:
                return lvl
        return ExitLevel.FULL


class Fixed:
    """Always the same level (baselines)."""
    def __init__(self, level):
        self.level = level

    def __call__(self, reading):
        return self.level


def policies(targets, probe: float, label_conf: float, seed: int) -> dict:
    """
    Every criterion compared in the experiments, by name:
    fixed levels, the paper's binary rule, the graded rule, learned UCB routers (one per target),
    a learned Thompson router at the middle target, and the hindsight oracle.
    """
    targets = list(targets)
    p = {
        "always_P3": Fixed(ExitLevel.P3),
        "always_P3_P4": Fixed(ExitLevel.P3_P4),
        "always_FULL": Fixed(ExitLevel.FULL),
        "heuristic_binary_30m": GradedRouter(full_dist=30, mid_dist=None, closing_rate=0.5),
        "heuristic_graded_30_60m": GradedRouter(full_dist=30, mid_dist=60, closing_rate=0.5),
    }
    for t in targets:
        p[f"learned_ucb_t{t:g}"] = LearnedRouter(target_miss=t, strategy="ucb", probe_rate=probe,
                                                 label_conf=label_conf, seed=seed)
    mid = targets[len(targets) // 2]
    p[f"learned_thompson_t{mid:g}"] = LearnedRouter(target_miss=mid, strategy="thompson", probe_rate=probe,
                                                   label_conf=label_conf, seed=seed)
    p["oracle"] = Oracle(label_conf, 0.5)
    return p


def replay(cache: dict, pols: dict, label_conf: float, own_vessel_bottom: float | None = None):
    lat = {ExitLevel[k]: v for k, v in cache["latency_ms"].items()}
    frames = cache["frames"]
    if own_vessel_bottom is not None:
        if not cache.get("frame_h"):
            raise SystemExit("cache has no frame height; re-run pass 1 to use --own-vessel-bottom")
        frames = [(r, drop_own_vessel(d, cache["frame_h"], own_vessel_bottom)) for r, d in frames]
    n = len(frames)
    # per-frame truth: misses of each level vs FULL
    truth = []
    for _, dets in frames:
        n_ref, m3 = count_missed(dets[ExitLevel.FULL], dets[ExitLevel.P3], 0.5, label_conf)
        _, m34 = count_missed(dets[ExitLevel.FULL], dets[ExitLevel.P3_P4], 0.5, label_conf)
        truth.append({ExitLevel.P3: m3, ExitLevel.P3_P4: m34, ExitLevel.FULL: 0, "n_ref": n_ref})

    summary, choices = [], {}
    for name, router in pols.items():
        if hasattr(router, "reset"):
            router.reset()
        levels, n_full_frames, n_probe = [], 0, 0
        for (reading, dets), tr in zip(frames, truth):
            if isinstance(router, Oracle):
                router.frame = dets
            lvl = router(reading)
            if isinstance(router, LearnedRouter):
                n_probe += router.last_probe
                if lvl == ExitLevel.FULL:
                    router.observe(dets)  # counterfactuals available only on FULL frames
            n_full_frames += lvl == ExitLevel.FULL
            levels.append(lvl)
        choices[name] = levels

        ref = sum(t["n_ref"] for t in truth)
        missed = sum(t[l] for t, l in zip(truth, levels))
        frames_with_ref = sum(t["n_ref"] > 0 for t in truth)
        frames_missed = sum(t[l] > 0 for t, l in zip(truth, levels))
        cost = sum(lat[l] for l in levels)
        summary.append({
            "policy": name,
            "speedup": lat[ExitLevel.FULL] * n / cost,
            "recall_vs_full": 1 - missed / ref if ref else float("nan"),
            "frame_miss": frames_missed / frames_with_ref if frames_with_ref else float("nan"),
            "P3": levels.count(ExitLevel.P3) / n,
            "P3_P4": levels.count(ExitLevel.P3_P4) / n,
            "FULL": n_full_frames / n,
            "probes": n_probe,
        })
        if isinstance(router, LearnedRouter):
            print(f"\n[{name}] learned weights (positive → more misses):\n{router.describe()}")
    return summary, choices, truth


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights")
    p.add_argument("--source")
    p.add_argument("--csv")
    p.add_argument("--mmsi", nargs=2, default=None)
    sync = p.add_mutually_exclusive_group()
    sync.add_argument("--video-start", default=None)
    sync.add_argument("--sync", default=None, metavar="VIDEO=LOG")
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--start", default=None, help="process from this video position (MM:SS)")
    p.add_argument("--end", default=None, help="stop at this video position (MM:SS)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default=None)
    p.add_argument("--profile-runs", type=int, default=100)
    p.add_argument("--out", default=None, help="output prefix (default: <source>_routers)")
    p.add_argument("--cache", default=None, help="reuse a pass-1 cache instead of running the model")
    p.add_argument("--reprofile", action="store_true",
                   help="with --cache and --weights: re-measure the per-level latency and store it in the cache")
    p.add_argument("--targets", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                   help="target miss rates for the learned routers")
    p.add_argument("--probe", type=float, default=0.05)
    p.add_argument("--label-conf", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--own-vessel-bottom", type=float, default=None, metavar="FRAC",
                   help="camera on a vessel: ignore boxes whose bottom edge is >= FRAC * frame height "
                        "(its own hull), e.g. 0.97")
    a = p.parse_args(argv)

    if a.cache:
        cache = pickle.loads(Path(a.cache).read_bytes())
        out = Path(a.out) if a.out else Path(a.cache).with_name(Path(a.cache).stem.replace("_cache", ""))
        if a.reprofile:
            if not a.weights:
                p.error("--reprofile needs --weights")
            prof = AdaptiveYOLO(a.weights, device=a.device, imgsz=a.imgsz).profile(runs=a.profile_runs)
            cache["latency_ms"] = {l.name: v for l, v in prof.items()}
            Path(a.cache).write_bytes(pickle.dumps(cache))
            print(f"[reprofile] latencies updated in {a.cache}")
    else:
        if not (a.weights and a.source and a.csv):
            p.error("--weights, --source and --csv are required unless --cache is given")
        out = Path(a.out) if a.out else Path(a.source).with_name(Path(a.source).stem + "_routers")
        cache = collect(a)
        cache_path = out.with_name(out.name + "_cache.pkl")
        cache_path.write_bytes(pickle.dumps(cache))
        print(f"[collect] cached → {cache_path}")

    summary, choices, truth = replay(cache, policies(a.targets, a.probe, a.label_conf, a.seed),
                                     a.label_conf, a.own_vessel_bottom)

    df = pd.DataFrame(summary).set_index("policy")
    print("\nlatency (ms, forward only): " + ", ".join(f"{k}={v:.2f}" for k, v in cache["latency_ms"].items()))
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    df.to_csv(out.with_name(out.name + "_summary.csv"))

    with open(out.with_name(out.name + "_frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "distance_m", "closure_rate_ms", "n_full_dets", "miss_P3", "miss_P3_P4", *choices])
        f0 = cache.get("start_frame", 0)
        for i, ((reading, _), t) in enumerate(zip(cache["frames"], truth)):
            w.writerow([f0 + i,f"{reading.distance_m:.2f}" if reading else "",
                        f"{reading.closure_rate_ms:.3f}" if reading else "",
                        t["n_ref"], t[ExitLevel.P3], t[ExitLevel.P3_P4],
                        *(choices[k][i].name for k in choices)])
    print(f"\n→ {out.name}_summary.csv, {out.name}_frames.csv")


if __name__ == "__main__":
    main()
