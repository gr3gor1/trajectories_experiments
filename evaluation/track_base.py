# """
# infer_video_full.py — YOLOv8 video inference, full model on every frame.
#
# All three detection heads (P3 + P4 + P5) run on every frame.
# No CSV, no proximity signals, no early-exit routing, no dual-path timing.
#
# Output video includes:
#   • Bounding boxes colour-coded by heuristic head attribution (size-based)
#   • HUD overlay: frame number, live FPS, total detections, per-head counts
#   • End-of-video summary card: timing, per-head detection bars, model metadata
# """
#
# import argparse
# import time
# from pathlib import Path
#
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from ultralytics import YOLO
# from ultralytics.nn.tasks import DetectionModel
# from ultralytics.nn.modules.conv import Conv
#
# torch.serialization.add_safe_globals([DetectionModel])
#
# # ── Constants ──────────────────────────────────────────────────────────────────
#
# NC      = 2
# CLASSES = {0: "asv", 1: "boat"}
# DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
#
# HEAD_NAMES  = {0: "P3 (small)", 1: "P4 (medium)", 2: "P5 (large)"}
# HEAD_ABBREV = {0: "P3", 1: "P4", 2: "P5"}
#
# HEAD_COLOUR = {
#     0: (255, 100,  50),
#     1: (50,  200, 255),
#     2: (80,  255, 140),
# }
#
# FONT       = cv2.FONT_HERSHEY_SIMPLEX
# FONT_SCALE = 0.55
# THICKNESS  = 2
#
#
# # ── Model loading ──────────────────────────────────────────────────────────────
#
# def _is_depthwise(state: dict) -> bool:
#     return "model.22.cv3.0.0.0.conv.weight" in state
#
#
# def _patch_cv3_depthwise(det: DetectionModel, state: dict) -> None:
#     detect = det.model[-1]
#     nc     = det.yaml["nc"]
#     for i in range(detect.nl):
#         dw_key = f"model.22.cv3.{i}.0.0.conv.weight"
#         C_in   = state[dw_key].shape[0]
#         new_cv3 = nn.Sequential(
#             nn.Sequential(Conv(C_in, C_in, 3, g=C_in), Conv(C_in, 64, 1)),
#             nn.Sequential(Conv(64,   64,   3, g=64),   Conv(64,  64, 1)),
#             nn.Conv2d(64, nc, 1),
#         )
#         detect.cv3[i] = new_cv3
#     print(f"[INFO] cv3 patched — depthwise-separable ({detect.nl} scales)")
#
#
# def load_model(weights: str, yaml_cfg: str = "yolov8n.yaml") -> YOLO:
#     raw = torch.load(weights, map_location=DEVICE, weights_only=False)
#
#     if isinstance(raw, dict) and "model" in raw and not any(
#         k.startswith("model.") for k in raw.keys()
#     ):
#         inner = raw["model"]
#         state = inner.state_dict() if isinstance(inner, nn.Module) else dict(inner)
#         print(f"[INFO] Training checkpoint  epoch={raw.get('epoch', '?')}")
#     else:
#         state = dict(raw)
#
#     state = {
#         k: v for k, v in state.items()
#         if not k.endswith((".total_ops", ".total_params"))
#     }
#
#     det = DetectionModel(yaml_cfg, nc=NC, verbose=False)
#
#     if _is_depthwise(state):
#         print("[INFO] Depthwise cv3 detected — patching before load")
#         _patch_cv3_depthwise(det, state)
#     else:
#         print("[INFO] Standard cv3 heads")
#
#     missing, _ = det.load_state_dict(state, strict=False)
#     real_missing = [k for k in missing if not k.endswith((".total_ops", ".total_params"))]
#     if real_missing:
#         raise RuntimeError(f"[ERROR] Missing weights:\n{real_missing}")
#
#     det.to(DEVICE).eval()
#     yolo       = YOLO(yaml_cfg)
#     yolo.model = det
#     return yolo
#
#
# # ── Detection drawing ──────────────────────────────────────────────────────────
#
# def attribute_to_head(xyxy: list, img_w: int, img_h: int) -> int:
#     """
#     Heuristic head attribution based on relative bounding-box area.
#     Small boxes  → P3, medium → P4, large → P5.
#     This is a display proxy only, not ground-truth head output.
#     """
#     bw  = (xyxy[2] - xyxy[0]) / img_w
#     bh  = (xyxy[3] - xyxy[1]) / img_h
#     rel = (bw * bh) ** 0.5
#     if rel < 0.15:
#         return 0
#     elif rel < 0.40:
#         return 1
#     return 2
#
#
# def draw_detections(frame: np.ndarray, results, img_w: int, img_h: int) -> tuple:
#     """Draw boxes from a model.predict() result. Returns (frame, head_counts)."""
#     frame_head_counts = {0: 0, 1: 0, 2: 0}
#     if results is None or len(results) == 0:
#         return frame, frame_head_counts
#
#     boxes = results[0].boxes
#     if boxes is None or len(boxes) == 0:
#         return frame, frame_head_counts
#
#     for box in boxes:
#         xyxy  = box.xyxy[0].tolist()
#         conf  = float(box.conf[0])
#         cls   = int(box.cls[0])
#         head  = attribute_to_head(xyxy, img_w, img_h)
#         color = HEAD_COLOUR[head]
#         name  = CLASSES.get(cls, str(cls))
#         x1, y1, x2, y2 = map(int, xyxy)
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
#
#         label = f"{HEAD_ABBREV[head]} {name} {conf:.2f}"
#         (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
#         cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
#                     FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
#
#         frame_head_counts[head] += 1
#
#     return frame, frame_head_counts
#
#
# # ── HUD overlay ────────────────────────────────────────────────────────────────
#
# def draw_hud(frame: np.ndarray, frame_no: int, total_head_counts: dict,
#              fps_live: float, total_inference_ms: float) -> np.ndarray:
#     total_det = sum(total_head_counts.values())
#     avg_ms    = total_inference_ms / max(frame_no, 1)
#
#     lines = [
#         f"Frame   : {frame_no}",
#         f"FPS     : {fps_live:.1f}",
#         f"Avg ms  : {avg_ms:.1f}",
#         f"Total   : {total_det}",
#         "",
#         *[f"  {HEAD_ABBREV[h]}  : {total_head_counts[h]}" for h in range(3)],
#     ]
#
#     panel_w = 210
#     panel_h = 22 + len(lines) * 18
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
#
#     for i, line in enumerate(lines):
#         col = (220, 220, 220)
#         cv2.putText(frame, line, (14, 26 + i * 18),
#                     FONT, 0.50, col, 1, cv2.LINE_AA)
#     return frame
#
#
# # ── Summary card ───────────────────────────────────────────────────────────────
#
# def build_summary_card(h: int, w: int, total_head_counts: dict,
#                        total_frames: int, duration_s: float,
#                        total_inference_ms: float, model_name: str) -> np.ndarray:
#     card = np.zeros((h, w, 3), dtype=np.uint8)
#     card[:] = (15, 15, 20)
#     cw = w
#
#     # Title
#     title = "INFERENCE SUMMARY"
#     (tw, _), _ = cv2.getTextSize(title, FONT, 1.3, 2)
#     cv2.putText(card, title, ((cw - tw) // 2, 65),
#                 FONT, 1.3, (240, 240, 240), 2, cv2.LINE_AA)
#     cv2.line(card, (cw // 8, 82), (7 * cw // 8, 82), (55, 55, 68), 1)
#
#     # Metadata strip
#     total_det = sum(total_head_counts.values())
#     avg_ms    = total_inference_ms / max(total_frames, 1)
#     meta_items = [
#         ("Model",      model_name),
#         ("Frames",     str(total_frames)),
#         ("Duration",   f"{duration_s:.1f}s"),
#         ("Detections", str(total_det)),
#     ]
#     col_w = cw // len(meta_items)
#     for i, (label, val) in enumerate(meta_items):
#         x = col_w * i + col_w // 2
#         cv2.putText(card, label, (x - 30, 114), FONT, 0.48, (120, 120, 140), 1, cv2.LINE_AA)
#         cv2.putText(card, val,   (x - 30, 132), FONT, 0.68, (230, 230, 230), 2, cv2.LINE_AA)
#     cv2.line(card, (cw // 8, 148), (7 * cw // 8, 148), (40, 40, 55), 1)
#
#     # Timing box
#     box_y = 170
#     bx0   = int(cw * 0.06)
#     bx1   = int(cw * 0.94)
#     cv2.rectangle(card, (bx0, box_y), (bx1, box_y + 100), (28, 28, 42), -1)
#     cv2.rectangle(card, (bx0, box_y), (bx1, box_y + 100), (60, 60, 82),  1)
#
#     cv2.putText(card, "FULL MODEL — TIMING",
#                 (bx0 + 18, box_y + 26), FONT, 0.68, (200, 200, 220), 1, cv2.LINE_AA)
#
#     timing_rows = [
#         ("Total inference time",
#          f"{total_inference_ms/1000:.2f}s",
#          (190, 190, 190)),
#         ("Average per frame",
#          f"{avg_ms:.2f} ms/frame",
#          (100, 255, 160)),
#         ("Average FPS (inference only)",
#          f"{1000/max(avg_ms, 1e-3):.1f}",
#          (100, 200, 255)),
#     ]
#     for i, (label, val, col) in enumerate(timing_rows):
#         cv2.putText(card, f"{label} : {val}",
#                     (bx0 + 18, box_y + 52 + i * 22),
#                     FONT, 0.52, col, 1, cv2.LINE_AA)
#
#     # Per-head bars
#     bar_y0    = box_y + 120
#     bar_x0    = int(cw * 0.22)
#     bar_max_w = int(cw * 0.52)
#     row_h     = 52
#
#     cv2.putText(card, "DETECTIONS  BY  HEAD  (heuristic attribution)",
#                 (cw // 2 - 230, bar_y0 - 8),
#                 FONT, 0.65, (180, 180, 200), 1, cv2.LINE_AA)
#
#     for idx in range(3):
#         count = total_head_counts.get(idx, 0)
#         color = HEAD_COLOUR[idx]
#         y     = bar_y0 + 14 + idx * row_h
#         pct   = count / max(total_det, 1) * 100
#         fill  = int(bar_max_w * count / max(total_det, 1))
#
#         cv2.putText(card, HEAD_NAMES[idx], (18, y + 22),
#                     FONT, 0.65, color, 2, cv2.LINE_AA)
#         cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + bar_max_w, y + 28),
#                       (38, 38, 50), -1)
#         if fill > 0:
#             cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + fill, y + 28), color, -1)
#         cv2.putText(card, f"{count}  ({pct:.1f}%)",
#                     (bar_x0 + bar_max_w + 12, y + 22),
#                     FONT, 0.60, (210, 210, 210), 1, cv2.LINE_AA)
#
#     note = "All frames processed with the full model (P3 + P4 + P5). No signal routing."
#     (fw, _), _ = cv2.getTextSize(note, FONT, 0.38, 1)
#     cv2.putText(card, note, ((cw - fw) // 2, h - 22),
#                 FONT, 0.38, (70, 70, 88), 1, cv2.LINE_AA)
#
#     return card
#
#
# # ── Main inference loop ────────────────────────────────────────────────────────
#
# def run(weights: str, source: str,
#         output: str = None,
#         conf: float = 0.25,
#         iou: float  = 0.45,
#         img_size: int = 640,
#         summary_frames: int = 90,
#         yaml_cfg: str = "yolov8n.yaml") -> str:
#
#     # 1. Load model
#     print(f"\n[INFO] Loading model : {weights}")
#     model = load_model(weights, yaml_cfg)
#
#     # 2. Open video and read metadata
#     cap = cv2.VideoCapture(source)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {source}")
#
#     src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     print(f"[INFO] Video         : {src_w}x{src_h}  {src_fps:.1f}fps  ~{total_f} frames")
#     print(f"[INFO] Device        : {DEVICE}")
#     print(f"[INFO] conf={conf}  iou={iou}  img_size={img_size}")
#
#     # 3. Prepare output writer
#     if output is None:
#         src_path = Path(source)
#         output   = str(src_path.parent / (src_path.stem + "_full_detected.mp4"))
#
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(output, fourcc, src_fps, (src_w, src_h))
#     if not writer.isOpened():
#         raise RuntimeError(f"Cannot open VideoWriter: {output}")
#
#     print(f"[INFO] Output        : {output}\n")
#
#     # 4. Frame loop
#     total_head_counts = {0: 0, 1: 0, 2: 0}
#     total_inference_ms = 0.0
#     frame_no           = 0
#     fps_live           = src_fps
#     t_wall_start       = time.perf_counter()
#
#     while True:
#         ret, bgr = cap.read()
#         if not ret:
#             break
#
#         frame_no += 1
#         frame = bgr.copy()
#
#         # Run full model and time it
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#         t0 = time.perf_counter()
#
#         results = model.predict(
#             frame,
#             imgsz   = img_size,
#             conf    = conf,
#             iou     = iou,
#             device  = DEVICE,
#             verbose = False,
#             classes = list(CLASSES.keys()),
#         )
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#         elapsed_ms = (time.perf_counter() - t0) * 1000
#
#         total_inference_ms += elapsed_ms
#         fps_live             = 1000.0 / max(elapsed_ms, 1e-3)
#
#         frame, frame_head_counts = draw_detections(frame, results, src_w, src_h)
#         for h in range(3):
#             total_head_counts[h] += frame_head_counts.get(h, 0)
#
#         frame = draw_hud(frame, frame_no, total_head_counts, fps_live, total_inference_ms)
#         writer.write(frame)
#
#         if frame_no % 50 == 0 or frame_no == 1:
#             pct = frame_no / max(total_f, 1) * 100
#             avg = total_inference_ms / frame_no
#             print(f"  [{frame_no:>6}/{total_f}]  {pct:5.1f}%  "
#                   f"fps={fps_live:5.1f}  avg={avg:.2f}ms  "
#                   f"heads={dict(total_head_counts)}")
#
#     duration_s = frame_no / src_fps
#     cap.release()
#
#     # 5. Summary card
#     print(f"\n[INFO] Appending summary card ({summary_frames} frames) ...")
#     card = build_summary_card(
#         src_h, src_w,
#         total_head_counts, frame_no, duration_s,
#         total_inference_ms, Path(weights).stem,
#     )
#     for _ in range(summary_frames):
#         writer.write(card)
#     writer.release()
#
#     # 6. Terminal report
#     wall  = time.perf_counter() - t_wall_start
#     avg   = total_inference_ms / max(frame_no, 1)
#     total = sum(total_head_counts.values())
#
#     print(f"\n{'='*60}")
#     print(f"  FINAL REPORT")
#     print(f"{'='*60}")
#     print(f"  Frames processed         : {frame_no}")
#     print(f"  Video duration           : {duration_s:.1f}s")
#     print(f"  Wall time (incl. I/O)    : {wall:.1f}s")
#     print()
#     print(f"  -- Inference timing ------")
#     print(f"  Total inference time     : {total_inference_ms/1000:.2f}s")
#     print(f"  Average per frame        : {avg:.2f} ms/frame")
#     print(f"  Average FPS (infer only) : {1000/max(avg,1e-3):.1f}")
#     print()
#     print(f"  -- Detections by head ----")
#     for h in range(3):
#         cnt = total_head_counts[h]
#         pct = cnt / max(total, 1) * 100
#         print(f"    {HEAD_NAMES[h]:15} : {cnt:6d}  ({pct:5.1f}%)")
#     print(f"    {'Total':15} : {total:6d}")
#     print()
#     print(f"  Output saved to          : {output}")
#     print(f"{'='*60}\n")
#
#     return output
#
#
# # ── Entry point ────────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
#     p = argparse.ArgumentParser(
#         description="YOLOv8 full-model video inference — all heads, every frame"
#     )
#     p.add_argument("--weights",       default="C:/Users/gr3g2/Downloads/nano_lr.pt",
#                    help="Path to .pt checkpoint")
#     p.add_argument("--source",         default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output.mp4",
#                    help="Path to input video")
#     p.add_argument("--output",         default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output_b.mp4",
#                    help="Path for output video (default: <source>_full_detected.mp4)")
#     p.add_argument("--conf",           type=float, default=0.25)
#     p.add_argument("--iou",            type=float, default=0.45)
#     p.add_argument("--img-size",       type=int,   default=640)
#     p.add_argument("--summary-frames", type=int,   default=90,
#                    help="Frames to hold on the summary card at end of video")
#     p.add_argument("--yaml",           default="yolov8n.yaml",
#                    help="Model architecture YAML (must match checkpoint)")
#     args = p.parse_args()
#
#     run(
#         weights        = args.weights,
#         source         = args.source,
#         output         = args.output,
#         conf           = args.conf,
#         iou            = args.iou,
#         img_size       = args.img_size,
#         summary_frames = args.summary_frames,
#         yaml_cfg       = args.yaml,
#     )

"""
infer_video.py — YOLOv8 video inference with proximity-signal head routing,
                 per-head attribution & rigorous speedup measurement.

Speedup measurement strategy (adopted from eval script):
  • Before the frame loop, FullForwardModel (backbone + neck + all 3 head
    branches, NO NMS/decode) and EarlyExitModel (P3 path only, NO NMS/decode)
    are each timed with N_WARMUP warmup runs followed by N_TIMING measured
    runs, using cuda.synchronize() bracketing. The MEDIAN is taken.
  • speedup = full_ms / p3_ms  — a clean latency ratio, not an accumulation.
  • Projected totals shown in summary/terminal are:
      projected_full_s  = full_ms  * total_frames / 1000
      projected_mixed_s = (full_ms  * n_full_frames
                           + p3_ms * n_p3_frames) / 1000
    These are honest projections from measured medians × frame counts.

HUD shows:
  • Fixed speedup ratio (from pre-measured profiling)
  • Live FPS of whichever path actually ran that frame

Signal routing (per frame, driven by AIS/GPS CSV):
  • FULL model — all three detection heads when objects are close AND closing.
  • P3 only    — P3 early-exit path when objects are far or not approaching.
"""

import argparse
import math
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.ops as tv_ops
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

torch.serialization.add_safe_globals([DetectionModel])

NC      = 2
CLASSES = {0: "asv", 1: "boat"}
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_NAMES  = {0: "P3 (small)", 1: "P4 (medium)", 2: "P5 (large)"}
HEAD_ABBREV = {0: "P3", 1: "P4", 2: "P5"}
HEAD_COLOUR = {
    0: (255, 100,  50),
    1: ( 50, 200, 255),
    2: ( 80, 255, 140),
}
HEAD_STRIDE = {0: 8, 1: 16, 2: 32}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

# ── Profiling constants (mirrors eval script) ──────────────────────────────────
N_WARMUP = 30
N_TIMING = 100

DEFAULT_DIST_THRESHOLD = 30.0
DEFAULT_RATE_THRESHOLD = 0.5

_R_EARTH = 6_371_000.0


# ══════════════════════════════════════════════════════════════════════════════
#  PROXIMITY SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lon1, lat1, lon2, lat2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * _R_EARTH * math.asin(math.sqrt(a))


def compute_proximity_signals(
    csv_path,
    video_fps=30.0,
    video_start_utc=None,
    dist_threshold=DEFAULT_DIST_THRESHOLD,
    rate_threshold=DEFAULT_RATE_THRESHOLD,
):
    df = pd.read_csv(csv_path)
    required = {"t", "mmsi", "lon", "lat"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["t"] = pd.to_datetime(df["t"], utc=True)
    df       = df.sort_values("t").reset_index(drop=True)

    mmsis = df["mmsi"].unique()
    if len(mmsis) < 2:
        raise ValueError(f"Expected 2 distinct MMSI values, found: {mmsis}")
    if len(mmsis) > 2:
        warnings.warn(f"More than 2 MMSI values ({mmsis}); using first two.", UserWarning)
        mmsis = mmsis[:2]

    mmsi_a, mmsi_b = int(mmsis[0]), int(mmsis[1])
    df_a = df[df["mmsi"] == mmsi_a][["t","lon","lat"]].drop_duplicates("t").set_index("t")
    df_b = df[df["mmsi"] == mmsi_b][["t","lon","lat"]].drop_duplicates("t").set_index("t")

    t_start = max(df_a.index.min(), df_b.index.min())
    t_end   = min(df_a.index.max(), df_b.index.max())
    if t_start >= t_end:
        raise ValueError("No overlapping timestamps between the two MMSI tracks.")

    grid = pd.date_range(t_start.ceil("s"), t_end.floor("s"), freq="1s", tz="UTC")
    if len(grid) == 0:
        grid = pd.DatetimeIndex([t_start.ceil("s")])

    def interp_to_grid(track, g):
        all_t = track.index.union(g).sort_values()
        return track.reindex(all_t).interpolate(method="time").reindex(g)

    ia = interp_to_grid(df_a, grid)
    ib = interp_to_grid(df_b, grid)

    valid = ~(ia["lon"].isna() | ia["lat"].isna() | ib["lon"].isna() | ib["lat"].isna())
    grid  = grid[valid]; ia = ia[valid]; ib = ib[valid]
    n     = len(grid)
    if n == 0:
        raise ValueError("No valid overlapping seconds after interpolation.")

    dists = np.array([
        _haversine(ia["lon"].iloc[i], ia["lat"].iloc[i],
                   ib["lon"].iloc[i], ib["lat"].iloc[i])
        for i in range(n)
    ])
    cr = np.concatenate([[np.diff(dists)[0]], np.diff(dists)]) if n > 1 else np.array([0.0])

    per_sec = np.where((dists < dist_threshold) & (cr < -rate_threshold), "FULL", "P3")

    vid_t0 = pd.Timestamp(video_start_utc, tz="UTC") if video_start_utc else t_start
    fps    = float(video_fps)
    frame_signals: dict[int, str] = {}
    for i, ts in enumerate(grid):
        off = (ts - vid_t0).total_seconds()
        f_s = max(0, math.floor(off * fps))
        f_e = math.floor((off + 1.0) * fps)
        for f in range(f_s, f_e):
            frame_signals[f] = per_sec[i]

    mid     = int(np.argmin(dists))
    summary = {
        "mmsi_a": mmsi_a, "mmsi_b": mmsi_b,
        "analysis_window_start": str(grid[0]),
        "analysis_window_end":   str(grid[-1]),
        "seconds_analysed":      n,
        "min_distance_m":        round(float(dists.min()), 2),
        "max_distance_m":        round(float(dists.max()), 2),
        "mean_distance_m":       round(float(dists.mean()), 2),
        "min_dist_at_second":    str(grid[mid]),
        "max_closing_rate_ms":   round(float(cr.min()), 3),
        "max_opening_rate_ms":   round(float(cr.max()), 3),
        "seconds_FULL":          int(np.sum(per_sec == "FULL")),
        "seconds_P3":            int(np.sum(per_sec == "P3")),
        "total_frames_mapped":   len(frame_signals),
        "frames_FULL":           sum(1 for v in frame_signals.values() if v == "FULL"),
        "frames_P3":             sum(1 for v in frame_signals.values() if v == "P3"),
        "dist_threshold_m":      dist_threshold,
        "rate_threshold_ms":     rate_threshold,
    }
    return frame_signals, summary


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _patch_cv3_depthwise(det, state):
    detect = det.model[-1]
    for i in range(detect.nl):
        dw_key = f"model.22.cv3.{i}.0.0.conv.weight"
        C_in   = state[dw_key].shape[0]
        new_cv3 = nn.Sequential(
            nn.Sequential(Conv(C_in, C_in, 3, g=C_in), Conv(C_in, 64, 1)),
            nn.Sequential(Conv(64,   64,   3, g=64),   Conv(64,  64, 1)),
            nn.Conv2d(64, NC, 1),
        )
        detect.cv3[i] = new_cv3
    print(f"[INFO] cv3 patched — depthwise-separable ({detect.nl} scales)")


def _is_depthwise(state):
    return "model.22.cv3.0.0.0.conv.weight" in state


def load_model(weights, yaml_cfg="yolov8n.yaml"):
    raw = torch.load(weights, map_location=DEVICE, weights_only=False)
    if isinstance(raw, dict) and "model" in raw and not any(
        k.startswith("model.") for k in raw.keys()
    ):
        inner = raw["model"]
        state = inner.state_dict() if isinstance(inner, nn.Module) else dict(inner)
        print(f"[INFO] Training checkpoint  epoch={raw.get('epoch','?')}")
    else:
        state = dict(raw)
    state = {k: v for k, v in state.items()
             if not k.endswith((".total_ops", ".total_params"))}

    det = DetectionModel(yaml_cfg, nc=NC, verbose=False)
    if _is_depthwise(state):
        print("[INFO] Depthwise cv3 detected — patching before load")
        _patch_cv3_depthwise(det, state)
    else:
        print("[INFO] Standard cv3 heads")

    missing, _ = det.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith((".total_ops", ".total_params"))]
    if real_missing:
        raise RuntimeError(f"[ERROR] Missing weights:\n{real_missing}")

    det.to(DEVICE).eval()
    yolo       = YOLO(yaml_cfg)
    yolo.model = det
    return yolo


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH TRACING & EARLY-EXIT MODULES
# ══════════════════════════════════════════════════════════════════════════════

def trace_head_path(layers, head_idx):
    detect_idx = len(layers) - 1
    det_f      = layers[detect_idx].f
    if isinstance(det_f, int):
        det_f = [det_f]
    det_f     = [detect_idx - 1 if f == -1 else f for f in det_f]
    head_root = det_f[head_idx]

    def get_parents(i):
        f = getattr(layers[i], "f", -1)
        if isinstance(f, int):
            return [i - 1 if f == -1 else f]
        return [i - 1 if fi == -1 else fi for fi in f]

    needed, queue = set(), [head_root]
    while queue:
        i = queue.pop()
        if i < 0 or i in needed:
            continue
        needed.add(i)
        queue.extend(get_parents(i))
    return sorted(needed)


class FullForwardModel(nn.Module):
    """
    Backbone + neck + ALL three cv2/cv3 head branches.
    No postprocessing / NMS.  Mirrors eval script's FullForwardModel exactly.
    This is the correct baseline: same logical boundary as EarlyExitModel.
    """
    def __init__(self, layers):
        super().__init__()
        self.backbone_neck = nn.ModuleList(layers[:-1])
        self.layer_f       = [getattr(l, "f", -1) for l in layers[:-1]]
        detect             = layers[-1]
        self.cv2           = detect.cv2
        self.cv3           = detect.cv3
        self.det_f         = detect.f

    def forward(self, x):
        n     = len(self.backbone_neck)
        cache = {}

        def resolve(fi, i):
            idx = i - 1 if fi == -1 else fi
            return x if idx < 0 else cache[idx]

        for i, layer in enumerate(self.backbone_neck):
            f   = self.layer_f[i]
            src = [resolve(fi, i) for fi in f] if isinstance(f, list) else resolve(f, i)
            cache[i] = layer(src) if isinstance(src, list) else layer(src)

        det_f = self.det_f
        if isinstance(det_f, int):
            det_f = [det_f]
        det_f = [n - 1 if fi == -1 else fi for fi in det_f]

        outputs = []
        for i, fi in enumerate(det_f):
            feat = cache[fi]
            outputs.append((self.cv2[i](feat), self.cv3[i](feat)))
        return outputs


class EarlyExitModel(nn.Module):
    """
    Partial backbone+neck (only layers needed for head_idx) + that head's
    cv2/cv3 branches.  No postprocessing / NMS.
    Scope matches FullForwardModel — speedup ratio is valid.
    """
    def __init__(self, layers, neck_indices, head_idx):
        super().__init__()
        self.exit      = max(neck_indices)
        self.needed    = set(neck_indices)
        self.layer_map = nn.ModuleDict({str(i): layers[i] for i in neck_indices})
        self.layer_f   = {i: getattr(layers[i], "f", -1) for i in neck_indices}
        detect         = layers[-1]
        self.cv2       = detect.cv2[head_idx]
        self.cv3       = detect.cv3[head_idx]

    def forward(self, x):
        cache = {}

        def resolve(fi, i):
            idx = i - 1 if fi == -1 else fi
            return x if idx < 0 else cache[idx]

        for i in range(self.exit + 1):
            if i not in self.needed:
                continue
            layer = self.layer_map[str(i)]
            f     = self.layer_f[i]
            src   = [resolve(fi, i) for fi in f] if isinstance(f, list) else resolve(f, i)
            cache[i] = layer(src) if isinstance(src, list) else layer(src)

        feat = cache[self.exit]
        return self.cv2(feat), self.cv3(feat)


# ══════════════════════════════════════════════════════════════════════════════
#  RIGOROUS LATENCY PROFILING  (adopted from eval script)
# ══════════════════════════════════════════════════════════════════════════════

def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def measure_time(module, dummy, n_warmup=N_WARMUP, n_runs=N_TIMING, label=""):
    """
    Warmup → synchronise → time N_TIMING individual forward passes →
    return the MEDIAN latency in milliseconds.

    Uses cuda.synchronize() before and after every individual run so that
    asynchronous kernel launches do not bleed across measurements.
    Deterministic=True / benchmark=False are set globally in run().
    """
    module.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            module(dummy)
        gpu_sync()

        times = []
        for _ in range(n_runs):
            gpu_sync()
            t0 = time.perf_counter()
            module(dummy)
            gpu_sync()
            times.append((time.perf_counter() - t0) * 1000)

    t      = torch.tensor(times)
    mean   = t.mean().item()
    std    = t.std().item()
    median = t.median().item()
    tag    = f"  [{label}]" if label else ""
    print(f"{tag}  mean={mean:.2f}ms  median={median:.2f}ms  "
          f"std={std:.2f}ms  cv={std/mean*100:.1f}%")
    return median


def build_latency_table(model, img_size=640):
    """
    Measure FullForwardModel and P3 EarlyExitModel using the same rigorous
    methodology as the eval script.  Returns:
        full_ms   — median latency of full forward pass (no NMS)
        p3_ms     — median latency of P3 early-exit (no NMS)
        speedup   — full_ms / p3_ms
    Both modules exclude NMS/decode so the ratio isolates backbone/neck cost.
    """
    layers = list(model.model.model)
    dummy  = torch.zeros(1, 3, img_size, img_size, device=DEVICE)

    # ── Full model ─────────────────────────────────────────────────────────────
    full_mod = FullForwardModel(layers).to(DEVICE)
    print(f"\n[PROFILE] Full forward model ({N_WARMUP} warmup, {N_TIMING} runs) …")
    full_ms = measure_time(full_mod, dummy, label="full")
    print(f"  → median {full_ms:.3f} ms/frame")
    del full_mod

    # ── P3 early-exit ──────────────────────────────────────────────────────────
    neck_idx = trace_head_path(layers, 0)
    ee_mod   = EarlyExitModel(layers, neck_idx, 0).to(DEVICE)
    print(f"\n[PROFILE] P3 early-exit ({N_WARMUP} warmup, {N_TIMING} runs) …")
    p3_ms = measure_time(ee_mod, dummy, label="P3")
    print(f"  → median {p3_ms:.3f} ms/frame")
    del ee_mod

    speedup = full_ms / p3_ms if p3_ms > 0 else 0.0

    print(f"\n{'─'*55}")
    print(f"  Latency comparison (backbone+neck+head, no NMS)")
    print(f"{'─'*55}")
    print(f"  Full model  : {full_ms:.3f} ms/frame")
    print(f"  P3 early-exit: {p3_ms:.3f} ms/frame")
    print(f"  Speedup     : {speedup:.2f}×")
    print(f"  (This ratio is used throughout; frame-loop I/O is not counted)")
    print(f"{'─'*55}\n")

    return full_ms, p3_ms, speedup


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME PRE-PROCESSING & P3 DECODE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_frame(bgr_frame, img_size, device):
    h0, w0 = bgr_frame.shape[:2]
    scale  = img_size / max(h0, w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))

    resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = img_size - nh, img_size - nw
    top, left    = pad_h // 2, pad_w // 2
    canvas       = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = resized

    # BGR (OpenCV) → RGB, matching what Ultralytics predict() feeds the full model
    tensor = (torch.from_numpy(canvas[..., ::-1].copy())
              .permute(2, 0, 1)
              .unsqueeze(0)
              .float()
              .to(device) / 255.0)
    return tensor, scale, left, top


def decode_p3(cv2_out, cv3_out, img_size, orig_w, orig_h,
              scale, pad_x, pad_y, conf_thresh, iou_thresh):
    stride = HEAD_STRIDE[0]
    device = cv2_out.device

    _, reg_ch, fh, fw = cv2_out.shape
    reg_max = reg_ch // 4
    nc      = cv3_out.shape[1]

    ys = torch.arange(fh, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(fw, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    anchors   = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    n_anchors = fh * fw

    box_logits = cv2_out[0].permute(1, 2, 0).reshape(n_anchors, 4, reg_max)
    box_dist   = box_logits.softmax(dim=-1)
    proj       = torch.arange(reg_max, device=device, dtype=torch.float32)
    box_dists  = (box_dist * proj).sum(dim=-1)

    lt, rb = box_dists[:, :2], box_dists[:, 2:]
    x1 = (anchors[:, 0] - lt[:, 0]) * stride
    y1 = (anchors[:, 1] - lt[:, 1]) * stride
    x2 = (anchors[:, 0] + rb[:, 0]) * stride
    y2 = (anchors[:, 1] + rb[:, 1]) * stride

    cls_logits = cv3_out[0].permute(1, 2, 0).reshape(n_anchors, nc)
    conf, cls  = cls_logits.sigmoid().max(dim=-1)

    mask = conf > conf_thresh
    if mask.sum() == 0:
        return []

    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    conf, cls       = conf[mask], cls[mask]

    boxes_t = torch.stack([x1, y1, x2, y2], dim=1)
    keep    = tv_ops.batched_nms(boxes_t, conf, cls, iou_thresh)
    boxes_t = boxes_t[keep].cpu()
    conf    = conf[keep].cpu()
    cls     = cls[keep].cpu()

    detections = []
    for (bx1, by1, bx2, by2), c, cl in zip(boxes_t.tolist(), conf.tolist(), cls.tolist()):
        ox1 = max(0, (bx1 - pad_x) / scale)
        oy1 = max(0, (by1 - pad_y) / scale)
        ox2 = min(orig_w, (bx2 - pad_x) / scale)
        oy2 = min(orig_h, (by2 - pad_y) / scale)
        if ox2 > ox1 and oy2 > oy1:
            detections.append((ox1, oy1, ox2, oy2, c, int(cl)))
    return detections


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION ATTRIBUTION & DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def attribute_to_head(xyxy, img_w, img_h):
    bw  = (xyxy[2] - xyxy[0]) / img_w
    bh  = (xyxy[3] - xyxy[1]) / img_h
    rel = (bw * bh) ** 0.5
    if rel < 0.15:   return 0
    elif rel < 0.40: return 1
    else:            return 2


def earliest_head_needed(frame_head_counts):
    for h in range(3):
        if frame_head_counts.get(h, 0) > 0:
            return h
    return 0


def draw_detections_from_results(frame, results, img_w, img_h):
    frame_head_counts = {0: 0, 1: 0, 2: 0}
    if results is None or len(results) == 0:
        return frame, frame_head_counts
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame, frame_head_counts

    for box in boxes:
        xyxy  = box.xyxy[0].tolist()
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        head  = attribute_to_head(xyxy, img_w, img_h)
        color = HEAD_COLOUR[head]
        name  = CLASSES.get(cls, str(cls))
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
        label = f"{HEAD_ABBREV[head]} {name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
        frame_head_counts[head] += 1

    return frame, frame_head_counts


def draw_detections_from_list(frame, detections, img_w, img_h):
    frame_head_counts = {0: 0, 1: 0, 2: 0}
    if not detections:
        return frame, frame_head_counts

    for (x1, y1, x2, y2, conf, cls) in detections:
        head  = 0
        color = HEAD_COLOUR[head]
        name  = CLASSES.get(cls, str(cls))
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, THICKNESS)
        label = f"{HEAD_ABBREV[head]} {name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (ix1, iy1 - th - bl - 4), (ix1 + tw + 4, iy1), color, -1)
        cv2.putText(frame, label, (ix1 + 2, iy1 - bl - 2),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
        frame_head_counts[head] += 1

    return frame, frame_head_counts


def draw_hud(frame, frame_no, total_head_counts, fps_live,
             active_head, signal, speedup):
    """
    HUD shows:
      • Signal (FULL / P3)
      • Live FPS of whichever path ran this frame
      • Fixed speedup ratio from pre-measured profiling (constant throughout)
    """
    total_det = sum(total_head_counts.values())

    lines = [
        f"Frame   : {frame_no}",
        f"Signal  : {signal}",
        f"FPS     : {fps_live:.1f}",
        f"Speedup : {speedup:.2f}x  (profiled)",
        f"Total   : {total_det}",
        "",
    ]
    for h in range(3):
        marker = " <" if h == active_head else "  "
        lines.append(f"  {HEAD_ABBREV[h]}{marker}: {total_head_counts[h]}")

    panel_w = 230
    panel_h = 22 + len(lines) * 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        if line.startswith("Signal"):
            col = (50, 140, 255) if signal == "FULL" else (80, 220, 80)
        elif line.startswith("Speedup"):
            col = (100, 200, 255)
        elif (line.startswith("  ") and
              line.strip().startswith(HEAD_ABBREV[active_head])):
            col = HEAD_COLOUR[active_head]
        else:
            col = (220, 220, 220)
        cv2.putText(frame, line, (14, 26 + i * 18),
                    FONT, 0.50, col, 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_card(h, w, total_head_counts, total_frames, duration_s,
                       full_ms, p3_ms, speedup,
                       n_full_frames, n_p3_frames,
                       model_name, prox_summary):
    """
    Projected totals = measured median × frame counts.
    Clearly labelled as projections, not accumulated wall-clock sums.
    """
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (15, 15, 20)
    cw = w

    # Title
    title = "INFERENCE SUMMARY"
    (tw, _), _ = cv2.getTextSize(title, FONT, 1.3, 2)
    cv2.putText(card, title, ((cw - tw) // 2, 65),
                FONT, 1.3, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.line(card, (cw // 8, 82), (7 * cw // 8, 82), (55, 55, 68), 1)

    # Metadata strip
    total_det  = sum(total_head_counts.values())
    meta_items = [
        ("Model",      model_name),
        ("Frames",     str(total_frames)),
        ("Duration",   f"{duration_s:.1f}s"),
        ("Detections", str(total_det)),
    ]
    col_w = cw // len(meta_items)
    for i, (label, val) in enumerate(meta_items):
        x = col_w * i + col_w // 2
        cv2.putText(card, label, (x - 30, 114), FONT, 0.48, (120, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(card, val,   (x - 30, 132), FONT, 0.68, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.line(card, (cw // 8, 148), (7 * cw // 8, 148), (40, 40, 55), 1)

    # Proximity routing stats
    sig_y = 170
    cv2.putText(card, "PROXIMITY  SIGNAL  ROUTING", (cw // 2 - 155, sig_y),
                FONT, 0.70, (180, 180, 200), 1, cv2.LINE_AA)
    prox_items = [
        (f"Min dist: {prox_summary['min_distance_m']:.1f} m",         (100, 255, 160)),
        (f"Max close rate: {abs(prox_summary['max_closing_rate_ms']):.2f} m/s", (100, 200, 255)),
        (f"FULL frames: {n_full_frames}",                              (50,  140, 255)),
        (f"P3-only frames: {n_p3_frames}",                             (80,  220,  80)),
    ]
    px = int(cw * 0.06)
    for i, (txt, col) in enumerate(prox_items):
        cv2.putText(card, txt,
                    (px + (i % 2) * (cw // 2), sig_y + 22 + (i // 2) * 20),
                    FONT, 0.52, col, 1, cv2.LINE_AA)
    cv2.line(card, (cw // 8, sig_y + 52), (7 * cw // 8, sig_y + 52), (40, 40, 55), 1)

    # Per-head detection bars
    bar_y0    = sig_y + 75
    bar_x0    = int(cw * 0.22)
    bar_max_w = int(cw * 0.52)
    row_h     = 56
    cv2.putText(card, "DETECTIONS  BY  HEAD", (cw // 2 - 145, bar_y0 - 8),
                FONT, 0.70, (180, 180, 200), 1, cv2.LINE_AA)

    for idx in range(3):
        count = total_head_counts.get(idx, 0)
        color = HEAD_COLOUR[idx]
        y     = bar_y0 + 14 + idx * row_h
        pct   = count / max(total_det, 1) * 100
        fill  = int(bar_max_w * count / max(total_det, 1))
        cv2.putText(card, HEAD_NAMES[idx], (18, y + 22),
                    FONT, 0.65, color, 2, cv2.LINE_AA)
        cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + bar_max_w, y + 28), (38, 38, 50), -1)
        if fill > 0:
            cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + fill, y + 28), color, -1)
        cv2.putText(card, f"{count}  ({pct:.1f}%)",
                    (bar_x0 + bar_max_w + 12, y + 22),
                    FONT, 0.60, (210, 210, 210), 1, cv2.LINE_AA)
        lat = full_ms if idx > 0 else p3_ms
        cv2.putText(card, f"profiled latency: {lat:.2f} ms/frame",
                    (bar_x0, y + 44), FONT, 0.42, (100, 100, 118), 1, cv2.LINE_AA)

    # ── Rigorous timing comparison box ─────────────────────────────────────────
    # Projected totals = measured median × frame count.
    # Clearly distinguished from raw wall-clock accumulation.
    box_y  = bar_y0 + 14 + 3 * row_h + 20
    box_h2 = 162
    bx0    = int(cw * 0.06)
    bx1    = int(cw * 0.94)
    cv2.rectangle(card, (bx0, box_y), (bx1, box_y + box_h2), (28, 28, 42), -1)
    cv2.rectangle(card, (bx0, box_y), (bx1, box_y + box_h2), (60, 60, 82),  1)

    cv2.putText(card, "RIGOROUS  LATENCY  COMPARISON",
                (bx0 + 18, box_y + 26),
                FONT, 0.68, (200, 200, 220), 1, cv2.LINE_AA)

    # Projected totals from profiled medians × frame counts
    proj_full_ms  = full_ms * total_frames
    proj_mixed_ms = full_ms * n_full_frames + p3_ms * n_p3_frames
    proj_full_s   = proj_full_ms  / 1000
    proj_mixed_s  = proj_mixed_ms / 1000
    saved_s       = proj_full_s - proj_mixed_s

    timing_rows = [
        ("Profiling method",
         f"{N_WARMUP} warmup + {N_TIMING} runs, cuda.sync, median",
         (140, 140, 160)),
        ("Full model latency",
         f"{full_ms:.3f} ms/frame  (backbone+neck+all heads, no NMS)",
         (190, 190, 190)),
        ("P3 early-exit latency",
         f"{p3_ms:.3f} ms/frame  (partial backbone+neck+P3 head, no NMS)",
         (190, 190, 190)),
        ("Per-frame speedup",
         f"{speedup:.2f}x  (full_ms / p3_ms)",
         (100, 200, 255)),
        ("Projected full total",
         f"{proj_full_s:.2f}s  [{full_ms:.2f} ms/frame × {total_frames} frames]",
         (190, 190, 190)),
        ("Projected mixed total",
         f"{proj_mixed_s:.2f}s  [FULL×{n_full_frames} + P3×{n_p3_frames} frames]",
         (100, 255, 160)),
        ("Projected time saved",
         f"{saved_s:.2f}s  (medians × frame counts — not wall-clock accumulation)",
         (100, 255, 160)),
    ]
    for i, (label, val, col) in enumerate(timing_rows):
        cv2.putText(card, f"{label} : {val}",
                    (bx0 + 18, box_y + 50 + i * 22),
                    FONT, 0.44, col, 1, cv2.LINE_AA)

    # Footer
    note = ("Speedup measured pre-loop: FullForwardModel vs EarlyExitModel, "
            "both exclude NMS/decode — apples-to-apples comparison.")
    (fw2, _), _ = cv2.getTextSize(note, FONT, 0.36, 1)
    cv2.putText(card, note, ((cw - fw2) // 2, h - 22),
                FONT, 0.36, (70, 70, 88), 1, cv2.LINE_AA)

    return card


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run(weights, source, csv_path,
        output=None, conf=0.25, iou=0.45,
        img_size=640, summary_frames=90, yaml_cfg="yolov8m.yaml",
        video_start_utc=None,
        dist_threshold=DEFAULT_DIST_THRESHOLD,
        rate_threshold=DEFAULT_RATE_THRESHOLD):

    # Deterministic GPU behaviour for reproducible profiling (mirrors eval script)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading model  : {weights}")
    model = load_model(weights, yaml_cfg)

    # ── 2. Build P3 early-exit module (kept alive for the whole run) ───────────
    layers   = list(model.model.model)
    neck_idx = trace_head_path(layers, 0)
    ee_p3    = EarlyExitModel(layers, neck_idx, 0).to(DEVICE).eval()
    print("[INFO] P3 early-exit module built and ready")

    # ── 3. Rigorous pre-loop profiling (script-2 methodology) ─────────────────
    #   Both FullForwardModel and EarlyExitModel are timed with:
    #     • N_WARMUP warmup passes (not counted)
    #     • N_TIMING timed passes with cuda.synchronize() before and after each
    #     • Median taken — robust to outlier spikes
    #   NMS/decode excluded from both sides → valid apples-to-apples ratio.
    full_ms, p3_ms, speedup = build_latency_table(model, img_size)

    # ── 4. Probe video metadata then load proximity signals ────────────────────
    cap_probe = cv2.VideoCapture(source)
    if not cap_probe.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    src_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    print(f"\n[INFO] Loading proximity CSV : {csv_path}")
    frame_signals, prox_summary = compute_proximity_signals(
        csv_path        = csv_path,
        video_fps       = src_fps,
        video_start_utc = video_start_utc,
        dist_threshold  = dist_threshold,
        rate_threshold  = rate_threshold,
    )
    print(f"[INFO] Proximity signals  : "
          f"{prox_summary['frames_FULL']} FULL frames, "
          f"{prox_summary['frames_P3']} P3-only frames "
          f"over {prox_summary['seconds_analysed']}s window")
    print(f"[INFO] Min distance       : {prox_summary['min_distance_m']} m")
    print(f"[INFO] Max closing rate   : {prox_summary['max_closing_rate_ms']} m/s")

    # ── 5. Open video & writer ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    print(f"\n[INFO] Video          : {src_w}x{src_h}  {src_fps:.1f}fps  ~{total_f} frames")

    if output is None:
        src_path = Path(source)
        output   = str(src_path.parent / (src_path.stem + "_detected.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output}")

    print(f"[INFO] Output         : {output}")
    print(f"[INFO] conf={conf}  iou={iou}  device={DEVICE}")
    print(f"[INFO] dist_threshold={dist_threshold}m  rate_threshold={rate_threshold}m/s")
    print(f"[INFO] Pre-profiled speedup: {speedup:.2f}x  "
          f"(full {full_ms:.2f}ms vs P3 {p3_ms:.2f}ms, no NMS)\n")

    # ── 6. Frame loop ──────────────────────────────────────────────────────────
    # The frame loop is inference-only.  No timing accumulation, no dual runs.
    # Speedup is already determined from step 3.
    # fps_live is measured per-frame for display only (not used for speedup).
    total_head_counts = {0: 0, 1: 0, 2: 0}
    signal_counts     = {"FULL": 0, "P3": 0}
    frame_no          = 0
    t_wall_start      = time.perf_counter()
    fps_live          = src_fps

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        frame_no += 1
        signal = frame_signals.get(frame_no - 1, "P3")
        signal_counts[signal] += 1
        frame = bgr.copy()

        if signal == "FULL":
            gpu_sync()
            t0 = time.perf_counter()
            results = model.predict(
                frame, imgsz=img_size, conf=conf, iou=iou,
                device=DEVICE, verbose=False, classes=list(CLASSES.keys()),
            )
            gpu_sync()
            fps_live = 1000.0 / max((time.perf_counter() - t0) * 1000, 1e-3)

            frame, frame_head_counts = draw_detections_from_results(
                frame, results, src_w, src_h)
            active_head = earliest_head_needed(frame_head_counts)

        else:
            gpu_sync()
            t0 = time.perf_counter()
            tensor, scale, pad_x, pad_y = preprocess_frame(frame, img_size, DEVICE)
            with torch.no_grad():
                cv2_out, cv3_out = ee_p3(tensor)
            detections = decode_p3(
                cv2_out, cv3_out, img_size, src_w, src_h,
                scale, pad_x, pad_y, conf, iou)
            gpu_sync()
            fps_live = 1000.0 / max((time.perf_counter() - t0) * 1000, 1e-3)

            frame, frame_head_counts = draw_detections_from_list(
                frame, detections, src_w, src_h)
            active_head = 0

        for h in range(3):
            total_head_counts[h] += frame_head_counts.get(h, 0)

        # HUD: fixed profiled speedup + live FPS of active path
        frame = draw_hud(frame, frame_no, total_head_counts, fps_live,
                         active_head, signal, speedup)
        writer.write(frame)

        if frame_no % 50 == 0 or frame_no == 1:
            pct = frame_no / max(total_f, 1) * 100
            print(f"  [{frame_no:>6}/{total_f}]  {pct:5.1f}%  "
                  f"signal={signal}  fps_live={fps_live:5.1f}  "
                  f"speedup={speedup:.2f}x  heads={dict(total_head_counts)}")

    duration_s = frame_no / src_fps
    cap.release()

    # ── 7. Summary card ────────────────────────────────────────────────────────
    print(f"\n[INFO] Appending summary card ({summary_frames} frames) ...")
    card = build_summary_card(
        src_h, src_w,
        total_head_counts, frame_no, duration_s,
        full_ms, p3_ms, speedup,
        signal_counts["FULL"], signal_counts["P3"],
        Path(weights).stem, prox_summary,
    )
    for _ in range(summary_frames):
        writer.write(card)
    writer.release()

    # ── 8. Terminal report ─────────────────────────────────────────────────────
    wall          = time.perf_counter() - t_wall_start
    proj_full_s   = full_ms * frame_no / 1000
    proj_mixed_s  = (full_ms * signal_counts["FULL"] + p3_ms * signal_counts["P3"]) / 1000
    saved_s       = proj_full_s - proj_mixed_s

    print(f"\n{'='*65}")
    print(f"  FINAL REPORT")
    print(f"{'='*65}")
    print(f"  Frames processed             : {frame_no}")
    print(f"  Video duration               : {duration_s:.1f}s")
    print(f"  Wall time (incl. I/O)        : {wall:.1f}s")
    print()
    print(f"  -- Proximity signal -----------")
    print(f"  Dist threshold               : {dist_threshold} m")
    print(f"  Rate threshold               : {rate_threshold} m/s")
    print(f"  Min distance observed        : {prox_summary['min_distance_m']} m")
    print(f"  Max closing rate             : {prox_summary['max_closing_rate_ms']} m/s")
    print(f"  Frames routed → FULL         : {signal_counts['FULL']}")
    print(f"  Frames routed → P3-only      : {signal_counts['P3']}")
    print()
    print(f"  -- Rigorous latency (pre-loop profiling) ------")
    print(f"  Method: {N_WARMUP} warmup + {N_TIMING} runs, cuda.sync, median, no NMS")
    print(f"  Full model latency           : {full_ms:.3f} ms/frame")
    print(f"  P3 early-exit latency        : {p3_ms:.3f} ms/frame")
    print(f"  Per-frame speedup            : {speedup:.2f}x")
    print()
    print(f"  -- Projected totals (median × frame count) ----")
    print(f"  Projected full total         : {proj_full_s:.2f}s")
    print(f"  Projected mixed total        : {proj_mixed_s:.2f}s")
    print(f"  Projected time saved         : {saved_s:.2f}s")
    print()
    print(f"  -- Detections by head ---------")
    total_det = sum(total_head_counts.values())
    for h in range(3):
        cnt = total_head_counts[h]
        pct = cnt / max(total_det, 1) * 100
        print(f"    {HEAD_NAMES[h]:15} : {cnt:6d}  ({pct:5.1f}%)")
    print(f"    {'Total':15} : {total_det:6d}")
    print()
    print(f"  Output saved to               : {output}")
    print(f"{'='*65}\n")

    return output, total_head_counts, full_ms, p3_ms, speedup


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="YOLOv8 inference with proximity-signal routing "
                    "and rigorous pre-loop speedup profiling"
    )
    p.add_argument("--weights",        default="C:/Users/gr3g2/Downloads/nano_lr.pt")
    p.add_argument("--source",         default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output.mp4")
    p.add_argument("--csv",            default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/ASV_logs_filtered.csv",
                   help="AIS/GPS CSV (columns: t, mmsi, lon, lat)")
    p.add_argument("--output",         default=None)
    p.add_argument("--conf",           type=float, default=0.25)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--img-size",       type=int,   default=640)
    p.add_argument("--summary-frames", type=int,   default=90)
    p.add_argument("--yaml",           default="yolov8n.yaml")
    p.add_argument("--video-start",    default=None,
                   help="UTC timestamp of frame 0 (ISO-8601)")
    p.add_argument("--dist",           type=float, default=DEFAULT_DIST_THRESHOLD,
                   help=f"Distance threshold in metres (default {DEFAULT_DIST_THRESHOLD})")
    p.add_argument("--rate",           type=float, default=DEFAULT_RATE_THRESHOLD,
                   help=f"Closing-rate threshold m/s (default {DEFAULT_RATE_THRESHOLD})")
    args = p.parse_args()

    run(
        weights         = args.weights,
        source          = args.source,
        csv_path        = args.csv,
        output          = args.output,
        conf            = args.conf,
        iou             = args.iou,
        img_size        = args.img_size,
        summary_frames  = args.summary_frames,
        yaml_cfg        = args.yaml,
        video_start_utc = args.video_start,
        dist_threshold  = args.dist,
        rate_threshold  = args.rate,
    )