"""
Run trajectory-guided adaptive YOLO on a recorded video + trajectory log.

    python -m adaptive_yolo.run_video --weights models/nano_lr.pt \
        --source output.mp4 --csv ASV_logs_simultaneous.csv \
        --video-start 2023-07-12T11:54:06

Outputs (next to --output):
    <name>.mp4         annotated video (boxes coloured by head, HUD with level/distance/rate)
    <name>_frames.csv  per frame: level, distance, closure rate, latency, detections per head

Use --force-level FULL to produce the always-full baseline with the same code path.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .learned import LearnedRouter, drop_own_vessel, is_own_vessel
from .model import AdaptiveYOLO, ExitLevel, HEAD_NAMES
from .router import GradedRouter
from .trajectory import TrajectoryLog, parse_video_position

HEAD_COLOUR = {0: (255, 100, 50), 1: (50, 200, 255), 2: (80, 255, 140)}  # BGR
LEVEL_COLOUR = {ExitLevel.P3: (255, 100, 50), ExitLevel.P3_P4: (50, 200, 255),
                ExitLevel.FULL: (60, 60, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def frame_range(start: str | None, end: str | None, fps: float, n_total: int) -> tuple[int, int]:
    """[start, end) frame indices from optional video positions (MM:SS, HH:MM:SS or seconds)."""
    start_f = round(parse_video_position(start) * fps) if start else 0
    end_f = min(round(parse_video_position(end) * fps), n_total) if end else n_total
    return start_f, end_f


def draw(frame, dets, names, level, reading, ms, scale: float = 1.0,
         src_h: int = 0, own_bottom: float | None = None):
    """Draw boxes (detections in source-pixel coords, scaled by `scale`) and the HUD."""
    for d in dets:
        p1 = (int(d.x1 * scale), int(d.y1 * scale))
        p2 = (int(d.x2 * scale), int(d.y2 * scale))
        own = is_own_vessel(d, src_h, own_bottom)
        col = (150, 150, 150) if own else HEAD_COLOUR[d.head]
        cv2.rectangle(frame, p1, p2, col, 2)
        label = ("own vessel (ignored) " if own else f"{names.get(d.cls, d.cls)} ") + \
            f"{d.conf:.2f} {HEAD_NAMES[d.head]}"
        cv2.putText(frame, label, (p1[0], max(12, p1[1] - 5)), FONT, 0.5, col, 1, cv2.LINE_AA)

    dist = f"{reading.distance_m:6.1f} m" if reading else "   n/a"
    rate = f"{reading.closure_rate_ms:+5.2f} m/s" if reading else "  n/a"
    lines = [f"level {level.name}", f"dist  {dist}", f"rate  {rate}", f"infer {ms:5.1f} ms"]
    cv2.rectangle(frame, (8, 8), (230, 22 + 22 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        col = LEVEL_COLOUR[level] if i == 0 else (230, 230, 230)
        cv2.putText(frame, t, (16, 32 + 22 * i), FONT, 0.6, col, 1, cv2.LINE_AA)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="input video")
    p.add_argument("--csv", help="trajectory log (t, mmsi, lon/lat or geom)")
    p.add_argument("--mmsi", nargs=2, default=None, help="the two vessel IDs to use from the log")
    sync = p.add_mutually_exclusive_group()
    sync.add_argument("--video-start", default=None,
                      help="UTC date+time of frame 0, e.g. 2023-07-12T11:54:06 "
                           "(default: the log's first record — right for a clip cut to match the log)")
    sync.add_argument("--sync", default=None, metavar="VIDEO=LOG",
                      help="pair a video position with a log time, e.g. 01:48=11:54:06 "
                           "(video MM:SS or HH:MM:SS or seconds; log HH:MM:SS on the log's date, same clock as column t)")
    p.add_argument("--offset", type=float, default=0.0,
                   help="extra seconds added to every frame's log time (fine-tuning; + = look later in the log)")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="abort if less than this fraction of the video has log data (0 to disable)")
    p.add_argument("--start", default=None, help="process from this video position (MM:SS, HH:MM:SS or s)")
    p.add_argument("--end", default=None, help="stop at this video position (default: end of video)")
    p.add_argument("--output", default=None, help="output .mp4 (default: <source>_adaptive.mp4)")
    p.add_argument("--out-width", type=int, default=None,
                   help="width of the annotated video (default: source width), e.g. 1920 for a 4K source")
    p.add_argument("--no-video", action="store_true", help="only write the per-frame CSV")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--rect", action="store_true", help="pad to stride multiple instead of square")
    p.add_argument("--device", default=None)
    p.add_argument("--full-dist", type=float, default=30.0)
    p.add_argument("--mid-dist", type=float, default=60.0,
                   help="P3+P4 below this distance; 0 = no middle level (the paper's binary P3/FULL rule)")
    p.add_argument("--rate", type=float, default=0.5, help="closing speed (m/s) required for FULL")
    p.add_argument("--hold", type=int, default=0, help="min frames before stepping down a level")
    p.add_argument("--router", choices=["graded", "learned"], default="graded",
                   help="graded = distance/closure thresholds; learned = online Bayesian router (learned.py)")
    p.add_argument("--target-miss", type=float, default=0.3,
                   help="learned: acceptable P(cheap level misses a FULL detection) per frame")
    p.add_argument("--strategy", choices=["ucb", "thompson", "greedy"], default="ucb")
    p.add_argument("--probe", type=float, default=0.02, help="learned: share of random FULL frames")
    p.add_argument("--force-level", choices=[l.name for l in ExitLevel], default=None,
                   help="ignore the trajectory and always use this level (baselines)")
    p.add_argument("--profile-runs", type=int, default=100, help="0 to skip latency profiling")
    p.add_argument("--own-vessel-bottom", type=float, default=None, metavar="FRAC",
                   help="camera on a vessel: boxes whose bottom edge is >= FRAC * frame height are its own "
                        "hull — not learned from, drawn grey (e.g. 0.97)")
    a = p.parse_args(argv)

    if a.csv is None and a.force_level is None:
        p.error("--csv is required unless --force-level is given")

    model = AdaptiveYOLO(a.weights, device=a.device, imgsz=a.imgsz, conf=a.conf, iou=a.iou, rect=a.rect)
    log = TrajectoryLog.from_csv(a.csv, *(a.mmsi or (None, None))) if a.csv else None
    if a.router == "learned":
        router = LearnedRouter(target_miss=a.target_miss, strategy=a.strategy, probe_rate=a.probe,
                               label_conf=a.conf)
    else:
        router = GradedRouter(a.full_dist, a.mid_dist or None, a.rate, a.hold)
    learning = isinstance(router, LearnedRouter) and a.force_level is None
    forced = ExitLevel[a.force_level] if a.force_level else None

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f, end_f = frame_range(a.start, a.end, fps, n_total)
    if not 0 <= start_f < end_f:
        raise SystemExit(f"Empty frame range {start_f}–{end_f} (video has {n_total} frames)")

    out_video = Path(a.output) if a.output else Path(a.source).with_name(Path(a.source).stem + "_adaptive.mp4")
    out_csv = out_video.with_name(out_video.stem + "_frames.csv")
    video_start = a.video_start
    if log is not None:
        n_ok = int(np.sum(~np.isnan(log.distance)))
        print(f"[log] MMSI {log.mmsi_a} vs {log.mmsi_b}: first record {log.first_record}, "
              f"{len(log.distance)} s ({n_ok} with both vessels), "
              f"distance {np.nanmin(log.distance):.1f}–{np.nanmax(log.distance):.1f} m")
        if a.sync:
            video_start = log.video_start_from_sync(a.sync)
        t0 = pd.Timestamp(video_start if video_start is not None else log.first_record)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        t0 = t0 + pd.Timedelta(seconds=start_f / fps)
        duration = (end_f - start_f) / fps
        cov = log.coverage(t0, duration, a.offset)
        t0 = t0 + pd.Timedelta(seconds=a.offset)
        print(f"[sync] frames {start_f}–{end_f - 1} ({duration:.1f} s) map to log {t0} → "
              f"{t0 + pd.Timedelta(seconds=duration)}; {100 * cov:.0f}% of it has trajectory data")
        if cov < a.min_coverage:
            raise SystemExit(
                f"Only {100 * cov:.0f}% of the video overlaps the log — the timestamps are not aligned.\n"
                f"Use --sync VIDEO=LOG (e.g. 00:00=11:54:06 for a clip whose first frame is 11:54:06 in the log),\n"
                f"check the camera/log timezone, or pass --min-coverage 0 to run anyway (uncovered frames run FULL).")

    out_w = a.out_width or w
    out_h = int(round(h * out_w / w / 2) * 2)
    writer = None if a.no_video else cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"Cannot read frame {start_f} of {a.source}")
    for _ in range(5):  # warm up the whole path (pre-process, every level, decode, NMS)
        for lvl in ExitLevel:
            model(first, lvl)
    model.sync()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    level_frames, level_ms, head_dets = Counter(), Counter(), Counter()
    with open(out_csv, "w", newline="") as fcsv:
        cw = csv.writer(fcsv)
        cw.writerow(["frame", "level", "distance_m", "closure_rate_ms", "latency_ms",
                     "n_det", "n_P3", "n_P4", "n_P5", "risk_P3", "risk_P3_P4", "probe"])
        f = start_f
        while f < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            reading = log.for_frame(f, fps, video_start, a.offset) if log else None
            level = forced if forced is not None else router(reading)

            model.sync()
            t0 = time.perf_counter()
            if learning and level == ExitLevel.FULL:
                # FULL frame: decode every level from the same forward pass and learn from it
                all_dets = model.detect_all_levels(frame)
                dets = all_dets[ExitLevel.FULL]
            else:
                all_dets, dets = None, model(frame, level)
            model.sync()
            ms = (time.perf_counter() - t0) * 1000
            if all_dets is not None:
                router.observe(drop_own_vessel(all_dets, h, a.own_vessel_bottom))

            per_head = Counter(d.head for d in dets)
            level_frames[level] += 1
            level_ms[level] += ms
            head_dets.update(per_head)
            cw.writerow([f, level.name,
                         f"{reading.distance_m:.2f}" if reading else "",
                         f"{reading.closure_rate_ms:.3f}" if reading else "",
                         f"{ms:.3f}", len(dets), per_head[0], per_head[1], per_head[2],
                         *(f"{router.last_risk[l]:.3f}" if learning and l in router.last_risk else ""
                           for l in (ExitLevel.P3, ExitLevel.P3_P4)),
                         int(router.last_probe) if learning else ""])
            if writer is not None:
                if out_w != w:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                draw(frame, dets, model.names, level, reading, ms, scale=out_w / w,
                     src_h=h, own_bottom=a.own_vessel_bottom)
                writer.write(frame)
            f += 1
            if (f - start_f) % 300 == 0:
                print(f"  frame {f}  " + "  ".join(f"{l.name}={level_frames[l]}" for l in ExitLevel))

    cap.release()
    if writer is not None:
        writer.release()

    n = sum(level_frames.values())
    print(f"\n{n} frames  →  {out_csv}" + ("" if a.no_video else f"\n            {out_video}"))
    print("\nlevel    frames   share   mean wall ms (incl. pre/post)")
    for l in ExitLevel:
        k = level_frames[l]
        print(f"{l.name:7s} {k:7d}  {100 * k / max(n, 1):5.1f}%   "
              f"{(level_ms[l] / k if k else float('nan')):8.2f}")
    print("detections by head: " + ", ".join(f"{HEAD_NAMES[h]}={head_dets[h]}" for h in range(3)))
    if learning:
        print(f"\nlearned router weights (positive → more misses):\n{router.describe()}")

    if a.profile_runs > 0 and n:
        prof = model.profile(runs=a.profile_runs)
        mixed = sum(prof[l] * level_frames[l] for l in ExitLevel)
        full = prof[ExitLevel.FULL] * n
        print("\nprofiled forward latency (median, no NMS): " +
              ", ".join(f"{l.name}={prof[l]:.2f} ms" for l in ExitLevel))
        print(f"projected: always-FULL {full / 1000:.2f} s, adaptive {mixed / 1000:.2f} s, "
              f"speedup {full / mixed:.2f}x, saved {100 * (1 - mixed / full):.1f}%")


if __name__ == "__main__":
    main()
