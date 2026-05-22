"""
infer_video.py — YOLOv8 video inference with per-head attribution & early-exit analysis.

For every frame the full model is run (all three detection heads).
After inference the script attributes each detection to the head that produced it,
accumulates per-head counts, and computes a *projected* total inference time:
"how long would the video have taken if we had stopped at the earliest head that
fired on each frame?"

Timing is honest and apples-to-apples:
  • full_projected_ms   — frame_no × full_ms_per_frame   (profiled forward-pass only)
  • cf_projected_ms     — Σ ee_ms[min_head]              (profiled forward-pass only)
  Both exclude preprocessing and NMS, so their ratio is a valid speedup.
  actual_inference_ms (wall-clock model.predict) is retained as an informational
  number only — it is NOT used in the speedup calculation.

End-of-video summary card shows:
  • Full-model projected inference time (profiled)
  • Per-head detection counts + share
  • Counterfactual time using the minimum-head strategy (profiled)
  • Projected speedup (both sides exclude pre/post — matched)
  • Wall-clock time shown separately as context

Usage:
    python infer_video.py --weights last.pt --source input.mp4
    python infer_video.py --weights last.pt --source input.mp4 \\
        --conf 0.35 --iou 0.5 --summary-frames 150
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

torch.serialization.add_safe_globals([DetectionModel])

# ── Config ─────────────────────────────────────────────────────────────────────
NC      = 2
CLASSES = {0: "asv", 1: "boat"}
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_NAMES  = {0: "P3 (small)", 1: "P4 (medium)", 2: "P5 (large)"}
HEAD_ABBREV = {0: "P3", 1: "P4", 2: "P5"}

# BGR colour per head
HEAD_COLOUR = {
    0: (255, 100,  50),   # P3 — blue
    1: (50,  200, 255),   # P4 — gold
    2: (80,  255, 140),   # P5 — mint
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

# Warmup & timing runs for early-exit latency profiling (done once at startup)
N_WARMUP = 20
N_TIMING = 60

# ── Architecture helpers ───────────────────────────────────────────────────────
def _patch_cv3_depthwise(det, state):
    detect = det.model[-1]
    nc     = det.yaml["nc"]
    for i in range(detect.nl):
        dw_key = f"model.22.cv3.{i}.0.0.conv.weight"
        C_in   = state[dw_key].shape[0]
        new_cv3 = nn.Sequential(
            nn.Sequential(Conv(C_in, C_in, 3, g=C_in), Conv(C_in, 64, 1)),
            nn.Sequential(Conv(64,   64,   3, g=64),   Conv(64,  64, 1)),
            nn.Conv2d(64, nc, 1),
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

# ── Graph tracing ──────────────────────────────────────────────────────────────
def trace_head_path(layers, head_idx):
    """BFS backwards to find backbone+neck layers needed by head_idx."""
    detect_idx = len(layers) - 1
    det_f      = layers[detect_idx].f
    if isinstance(det_f, int):
        det_f = [det_f]
    det_f = [detect_idx - 1 if f == -1 else f for f in det_f]
    head_root = det_f[head_idx]

    def get_parents(i):
        f = getattr(layers[i], 'f', -1)
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

# ── Early-exit model ───────────────────────────────────────────────────────────
class EarlyExitModel(nn.Module):
    def __init__(self, layers, neck_indices, head_idx):
        super().__init__()
        self.exit      = max(neck_indices)
        self.head      = head_idx
        self.needed    = set(neck_indices)
        self.layer_map = nn.ModuleDict({str(i): layers[i] for i in neck_indices})
        self.layer_f   = {i: getattr(layers[i], 'f', -1) for i in neck_indices}
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

# ── Latency profiling ──────────────────────────────────────────────────────────
def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def profile_latency(module, dummy, n_warmup=N_WARMUP, n_runs=N_TIMING):
    """Returns median latency in milliseconds (forward-pass only, no pre/post)."""
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
    t  = torch.tensor(times)
    ms = t.median().item()
    print(f"  median={ms:.3f}ms  std={t.std().item():.3f}ms")
    return ms


def build_latency_table(model, img_size=640):
    """
    Profiles full model + each early-exit path once.

    Both full_ms and ee_ms are pure forward-pass numbers (no preprocessing,
    no NMS). The speedup ratio full_ms / ee_ms[h] is therefore valid because
    both sides exclude the same stages.

    Returns:
        full_ms : float            — median full-model forward-pass latency (ms)
        ee_ms   : dict[int, float] — median early-exit forward-pass latency per head (ms)
    """
    layers = list(model.model.model)
    dummy  = torch.zeros(1, 3, img_size, img_size, device=DEVICE)

    print("\n[PROFILE] Full model (forward-pass only) …")
    full_ms = profile_latency(model.model, dummy)
    print(f"  → {full_ms:.3f} ms/frame")

    ee_ms = {}
    for h in range(3):
        neck_idx = trace_head_path(layers, h)
        ee_mod   = EarlyExitModel(layers, neck_idx, h).to(DEVICE)
        print(f"[PROFILE] Early-exit head {h} ({HEAD_NAMES[h]}) (forward-pass only) …")
        ee_ms[h] = profile_latency(ee_mod, dummy)
        print(f"  → {ee_ms[h]:.3f} ms/frame  ({ee_ms[h]/full_ms*100:.1f}% of full)")
        del ee_mod

    return full_ms, ee_ms

# ── Per-head detection attribution ────────────────────────────────────────────
def attribute_to_head(xyxy, img_w, img_h):
    """
    Assigns a detection box to the most likely head based on its relative size.

    YOLOv8 head design intent:
      P3 (stride 8)  — small objects   rel_size < 0.15
      P4 (stride 16) — medium objects  0.15 ≤ rel_size < 0.40
      P5 (stride 32) — large objects   rel_size ≥ 0.40

    rel_size = geometric mean of (box_w/img_w, box_h/img_h)

    NOTE: this is a heuristic approximation. A detection that appears near
    a threshold boundary could have come from either adjacent head. The
    attribution affects which ee_ms value is accumulated in cf_projected_ms,
    so boundary-zone inaccuracies slightly bias the projected speedup.
    """
    bw  = (xyxy[2] - xyxy[0]) / img_w
    bh  = (xyxy[3] - xyxy[1]) / img_h
    rel = (bw * bh) ** 0.5

    if rel < 0.15:
        return 0
    elif rel < 0.40:
        return 1
    else:
        return 2

# ── Per-frame minimum-head strategy ───────────────────────────────────────────
def earliest_head_needed(frame_head_counts):
    """
    Returns the index of the lowest-numbered head that fired this frame.
    Falls back to head 0 (P3, cheapest) for empty frames.

    NOTE: using P3 for empty frames slightly underestimates cf_projected_ms
    because in a real early-exit pipeline an empty frame still runs all heads
    before concluding nothing was found. Document this assumption.
    """
    for h in range(3):
        if frame_head_counts.get(h, 0) > 0:
            return h
    return 0

# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_detections(frame, results, img_w, img_h):
    """
    Draws boxes colour-coded by attributed head.
    Returns (annotated_frame, {head_idx: count_this_frame}).
    """
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


def draw_hud(frame, frame_no, total_head_counts, fps_live,
             active_head_this_frame, full_ms_per_frame, cf_ms_so_far, frames_so_far):
    """
    HUD top-left: frame counter, live FPS, running per-head totals,
    and a running projected speedup (both sides forward-pass only).
    """
    full_proj_so_far = frames_so_far * full_ms_per_frame
    speedup_live     = (full_proj_so_far / cf_ms_so_far
                        if cf_ms_so_far > 0 else 1.0)
    total_det        = sum(total_head_counts.values())

    lines = [
        f"Frame   : {frame_no}",
        f"FPS     : {fps_live:.1f}",
        f"Speedup : {speedup_live:.2f}x (proj)",
        f"Total   : {total_det}",
        "",
    ]
    for h in range(3):
        marker = " ◀" if h == active_head_this_frame else "  "
        lines.append(f"  {HEAD_ABBREV[h]}{marker}: {total_head_counts[h]}")

    panel_w = 210
    panel_h = 22 + len(lines) * 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        if line.startswith("Speedup"):
            col = (100, 200, 255)
        elif (line.startswith("  ") and
              line.strip().startswith(HEAD_ABBREV[active_head_this_frame])):
            col = HEAD_COLOUR[active_head_this_frame]
        else:
            col = (220, 220, 220)
        cv2.putText(frame, line, (14, 26 + i * 18),
                    FONT, 0.50, col, 1, cv2.LINE_AA)

    return frame

# ── Summary card ───────────────────────────────────────────────────────────────
def build_summary_card(h, w, total_head_counts, total_frames, duration_s,
                       full_ms_per_frame, ee_ms,
                       cf_projected_ms, actual_inference_ms,
                       model_name):
    """
    Timing displayed on this card:
      full_projected_ms = total_frames × full_ms_per_frame  (forward-pass profile)
      cf_projected_ms   = Σ ee_ms[min_head per frame]       (forward-pass profile)
      speedup           = full_projected_ms / cf_projected_ms   ← apples-to-apples
      actual_inference_ms is shown as context only (includes pre/post).
    """
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (15, 15, 20)

    full_projected_ms = total_frames * full_ms_per_frame
    total_det         = sum(total_head_counts.values())
    cw                = w

    # ── Title ──────────────────────────────────────────────────────────
    title = "INFERENCE SUMMARY"
    (tw, _), _ = cv2.getTextSize(title, FONT, 1.3, 2)
    cv2.putText(card, title, ((cw - tw) // 2, 65),
                FONT, 1.3, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.line(card, (cw // 8, 82), (7 * cw // 8, 82), (55, 55, 68), 1)

    # ── Metadata strip ─────────────────────────────────────────────────
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

    # ── Per-head bars ──────────────────────────────────────────────────
    cv2.putText(card, "DETECTIONS  BY  HEAD", (cw // 2 - 145, 175),
                FONT, 0.70, (180, 180, 200), 1, cv2.LINE_AA)

    bar_x0    = int(cw * 0.22)
    bar_max_w = int(cw * 0.52)
    row_h     = 56
    y0        = 198

    for idx in range(3):
        count = total_head_counts.get(idx, 0)
        color = HEAD_COLOUR[idx]
        y     = y0 + idx * row_h
        pct   = count / max(total_det, 1) * 100
        fill  = int(bar_max_w * count / max(total_det, 1))

        cv2.putText(card, HEAD_NAMES[idx], (18, y + 22),
                    FONT, 0.65, color, 2, cv2.LINE_AA)

        cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + bar_max_w, y + 28),
                      (38, 38, 50), -1)
        if fill > 0:
            cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + fill, y + 28), color, -1)

        tag = f"{count}  ({pct:.1f}%)"
        cv2.putText(card, tag, (bar_x0 + bar_max_w + 12, y + 22),
                    FONT, 0.60, (210, 210, 210), 1, cv2.LINE_AA)

        ee_note = f"profiled fwd-pass latency: {ee_ms[idx]:.2f} ms/frame"
        cv2.putText(card, ee_note, (bar_x0, y + 44),
                    FONT, 0.42, (100, 100, 118), 1, cv2.LINE_AA)

    # ── Timing comparison box ──────────────────────────────────────────
    box_y  = y0 + 3 * row_h + 20
    box_h2 = 152
    box_x0 = int(cw * 0.06)
    box_x1 = int(cw * 0.94)
    cv2.rectangle(card, (box_x0, box_y), (box_x1, box_y + box_h2), (28, 28, 42), -1)
    cv2.rectangle(card, (box_x0, box_y), (box_x1, box_y + box_h2), (60, 60, 82), 1)

    cv2.putText(card, "PROJECTED  INFERENCE  TIME  (forward-pass only)",
                (box_x0 + 18, box_y + 26),
                FONT, 0.65, (200, 200, 220), 1, cv2.LINE_AA)

    full_proj_s = full_projected_ms / 1000
    cf_s        = cf_projected_ms / 1000
    saved_s     = full_proj_s - cf_s
    speedup     = full_proj_s / cf_s if cf_s > 0 else 0
    wall_s      = actual_inference_ms / 1000

    timing_rows = [
        (f"Full model  (profiled fwd × {total_frames} frames) : "
         f"{full_proj_s:.2f}s  [{full_ms_per_frame:.2f} ms/frame]",
         (190, 190, 190)),
        (f"Early-exit  (min-head, profiled fwd)       : "
         f"{cf_s:.2f}s",
         (100, 255, 160)),
        (f"Time saved  |  Projected speedup            : "
         f"{saved_s:.2f}s  |  {speedup:.2f}x",
         (100, 200, 255)),
        (f"Wall-clock  (incl. pre/post, context only)  : "
         f"{wall_s:.2f}s",
         (130, 130, 150)),
    ]
    for i, (txt, col) in enumerate(timing_rows):
        cv2.putText(card, txt, (box_x0 + 18, box_y + 54 + i * 26),
                    FONT, 0.50, col, 1, cv2.LINE_AA)

    # ── Footer note ────────────────────────────────────────────────────
    note = ("Projected speedup uses matched forward-pass profiles only (no pre/post on either side). "
            "Attribution uses rel-size heuristic; empty frames assumed cheapest head (P3).")
    (fw, _), _ = cv2.getTextSize(note, FONT, 0.36, 1)
    cv2.putText(card, note, ((cw - fw) // 2, h - 22),
                FONT, 0.36, (70, 70, 88), 1, cv2.LINE_AA)

    return card

# ── Main inference loop ────────────────────────────────────────────────────────
def run(weights, source, output=None, conf=0.25, iou=0.45,
        img_size=640, summary_frames=90, yaml_cfg="yolov8m.yaml"):

    print(f"\n[INFO] Loading model  : {weights}")
    model = load_model(weights, yaml_cfg)

    # Profile latencies once before processing.
    # These are pure forward-pass numbers — no preprocessing, no NMS.
    # full_ms_per_frame and ee_ms are on the same footing, so their
    # ratio is a valid speedup estimate.
    full_ms_per_frame, ee_ms = build_latency_table(model, img_size)

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[INFO] Video          : {src_w}x{src_h}  {src_fps:.1f}fps  ~{total_f} frames")
    print("[INFO] Timing note    : speedup is projected from matched fwd-pass profiles.")
    print("                        Wall-clock (model.predict) is logged but NOT used")
    print("                        in the speedup ratio (it includes pre/post; profiles don't).\n")

    if output is None:
        src_path = Path(source)
        output   = str(src_path.parent / (src_path.stem + "_detected.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output}")

    print(f"[INFO] Output         : {output}")
    print(f"[INFO] conf={conf}  iou={iou}  device={DEVICE}\n")

    # Accumulators
    total_head_counts   = {0: 0, 1: 0, 2: 0}

    # cf_projected_ms: Σ ee_ms[min_head] — forward-pass profile, matches full_ms_per_frame
    cf_projected_ms     = 0.0

    # actual_inference_ms: wall-clock model.predict() — includes pre/post.
    # Kept for informational display only; NOT used in speedup.
    actual_inference_ms = 0.0

    frame_no         = 0
    t_wall_start     = time.perf_counter()
    fps_live         = src_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        gpu_sync()
        t0 = time.perf_counter()

        results = model.predict(
            frame,
            imgsz   = img_size,
            conf    = conf,
            iou     = iou,
            device  = DEVICE,
            verbose = False,
            classes = list(CLASSES.keys()),
        )

        gpu_sync()
        elapsed_ms           = (time.perf_counter() - t0) * 1000
        fps_live             = 1000.0 / max(elapsed_ms, 1e-3)
        actual_inference_ms += elapsed_ms   # informational only

        # Attribute each detection to a head and draw
        frame, frame_head_counts = draw_detections(frame, results, src_w, src_h)
        for h in range(3):
            total_head_counts[h] += frame_head_counts.get(h, 0)

        # Counterfactual: which head would have been sufficient?
        # Accumulate its profiled forward-pass latency (same units as full_ms_per_frame).
        min_head         = earliest_head_needed(frame_head_counts)
        cf_projected_ms += ee_ms[min_head]

        frame = draw_hud(frame, frame_no, total_head_counts, fps_live,
                         min_head, full_ms_per_frame, cf_projected_ms, frame_no)
        writer.write(frame)

        if frame_no % 50 == 0 or frame_no == 1:
            pct              = frame_no / max(total_f, 1) * 100
            full_proj_so_far = frame_no * full_ms_per_frame
            spd              = full_proj_so_far / cf_projected_ms if cf_projected_ms > 0 else 1.0
            print(f"  [{frame_no:>6}/{total_f}]  {pct:5.1f}%  "
                  f"wall_fps={fps_live:5.1f}  "
                  f"proj_speedup={spd:.2f}x  "
                  f"heads={dict(total_head_counts)}")

    duration_s = frame_no / src_fps
    cap.release()

    # Append summary card
    print(f"\n[INFO] Appending summary card ({summary_frames} frames) ...")
    card = build_summary_card(
        src_h, src_w,
        total_head_counts,
        frame_no,
        duration_s,
        full_ms_per_frame,
        ee_ms,
        cf_projected_ms,
        actual_inference_ms,
        Path(weights).stem,
    )
    for _ in range(summary_frames):
        writer.write(card)
    writer.release()

    # ── Terminal report ────────────────────────────────────────────────────────
    wall              = time.perf_counter() - t_wall_start
    full_projected_ms = frame_no * full_ms_per_frame
    full_proj_s       = full_projected_ms / 1000
    cf_s              = cf_projected_ms / 1000
    saved_s           = full_proj_s - cf_s
    speedup           = full_proj_s / cf_s if cf_s > 0 else 0
    wall_infer_s      = actual_inference_ms / 1000

    print(f"\n{'='*65}")
    print(f"  FINAL REPORT")
    print(f"{'='*65}")
    print(f"  Frames processed              : {frame_no}")
    print(f"  Video duration                : {duration_s:.1f}s")
    print(f"  Wall time (incl. I/O)         : {wall:.1f}s")
    print()
    print(f"  -- Projected inference time (forward-pass only) --")
    print(f"  (Both sides profiled without preprocessing or NMS)")
    print(f"  Full model   ({full_ms_per_frame:.2f} ms/frame × {frame_no} frames) "
          f": {full_proj_s:.2f}s")
    print(f"  Early-exit   (min-head Σ)                      "
          f": {cf_s:.2f}s")
    print(f"  Time saved                                       "
          f": {saved_s:.2f}s")
    print(f"  Projected speedup                                "
          f": {speedup:.2f}x")
    print()
    print(f"  -- Wall-clock (informational only) ---------------")
    print(f"  model.predict() total (incl. pre/post)           "
          f": {wall_infer_s:.2f}s")
    print(f"  Avg per frame                                    "
          f": {actual_inference_ms/max(frame_no,1):.2f} ms")
    print(f"  NOTE: this is NOT used in the speedup ratio.")
    print(f"        It includes letterbox, normalize, NMS, decode")
    print(f"        which the profiled ee_ms values do not include.")
    print()
    print(f"  -- Detections by head ---")
    total_det = sum(total_head_counts.values())
    for h in range(3):
        cnt = total_head_counts[h]
        pct = cnt / max(total_det, 1) * 100
        print(f"    {HEAD_NAMES[h]:15} : {cnt:6d}  ({pct:5.1f}%)  "
              f"ee_fwd={ee_ms[h]:.2f}ms/frame")
    print(f"    {'Total':15} : {total_det:6d}")
    print()
    print(f"  -- Attribution notes ---")
    print(f"  Head assigned by relative box size (heuristic).")
    print(f"  Empty frames assume P3 (cheapest head) — slightly")
    print(f"  underestimates cf_projected_ms for truly empty frames.")
    print()
    print(f"  Output saved to               : {output}")
    print(f"{'='*65}\n")

    return output, total_head_counts, actual_inference_ms, cf_projected_ms

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="C:/Users/gr3g2/Downloads/nano_lr.pt",
                   help="Path to .pt checkpoint")
    p.add_argument("--source",   default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output.mp4",
                   help="Input video path")
    p.add_argument("--output",         default=None,
                   help="Output path (default: <source>_detected.mp4)")
    p.add_argument("--conf",           type=float, default=0.25)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--img-size",       type=int,   default=640)
    p.add_argument("--summary-frames", type=int,   default=90,
                   help="Frames to hold summary card (default 90 = ~3s @ 30fps)")
    p.add_argument("--yaml",           default="yolov8n.yaml")
    args = p.parse_args()

    run(
        weights        = args.weights,
        source         = args.source,
        output         = args.output,
        conf           = args.conf,
        iou            = args.iou,
        img_size       = args.img_size,
        summary_frames = args.summary_frames,
        yaml_cfg       = args.yaml,
    )