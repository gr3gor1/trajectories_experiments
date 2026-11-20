"""
infer_compare.py — Reliable full-model vs P3-only comparison.

Both paths run on EVERY frame regardless of any proximity signal.
Results are tracked completely independently so comparisons are fair.

Per frame, the script measures:
  • Wall-clock inference time for the full model
  • Wall-clock inference time for the P3 early-exit path
  • Detections from each path: bounding boxes, class IDs, confidence scores,
    and size-bucket attribution (small / medium / large)

End-of-run CSV reports:
  • per_frame_timing.csv      — frame-by-frame ms for both paths
  • full_detections.csv       — every box from the full model (all frames)
  • p3_detections.csv         — every box from the P3 path (all frames)
  • summary.csv               — aggregate timing, counts, class & size breakdown

Output video shows both paths side-by-side on a split frame with a shared HUD.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as tv_ops
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

torch.serialization.add_safe_globals([DetectionModel])

# ── Constants ──────────────────────────────────────────────────────────────────

NC      = 2
CLASSES = {0: "asv", 1: "boat"}
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# Size buckets — same thresholds as original attribute_to_head()
SIZE_LABELS  = {0: "small", 1: "medium", 2: "large"}
SIZE_THRESHS = (0.15, 0.40)          # relative side-length thresholds

HEAD_STRIDE  = {0: 8, 1: 16, 2: 32}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50
THICKNESS  = 1

# Colours: full model = blue family, P3 = orange family, per size bucket
COLOUR_FULL = {0: (255, 200,  80), 1: (80,  200, 255), 2: (80,  255, 140)}
COLOUR_P3   = {0: (255,  80,  50), 1: (200,  80, 255), 2: (255, 180,  50)}


# ── Model loading (unchanged from original) ────────────────────────────────────

def _is_depthwise(state):
    return "model.22.cv3.0.0.0.conv.weight" in state


def _patch_cv3_depthwise(det, state):
    detect = det.model[-1]
    for i in range(detect.nl):
        dw_key = f"model.22.cv3.{i}.0.0.conv.weight"
        C_in   = state[dw_key].shape[0]
        detect.cv3[i] = nn.Sequential(
            nn.Sequential(Conv(C_in, C_in, 3, g=C_in), Conv(C_in, 64, 1)),
            nn.Sequential(Conv(64,   64,   3, g=64),   Conv(64,  64, 1)),
            nn.Conv2d(64, NC, 1),
        )
    print(f"[INFO] cv3 patched — depthwise-separable ({detect.nl} scales)")


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


# ── P3 early-exit (unchanged from original) ────────────────────────────────────

def trace_head_path(layers, head_idx):
    detect_idx = len(layers) - 1
    det_f = layers[detect_idx].f
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


class EarlyExitModel(nn.Module):
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


def preprocess_frame(bgr_frame, img_size, device):
    h0, w0 = bgr_frame.shape[:2]
    scale  = img_size / max(h0, w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = img_size - nh, img_size - nw
    top, left    = pad_h // 2, pad_w // 2
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = resized
    # BGR (OpenCV) → RGB, matching what Ultralytics predict() feeds the full model
    tensor = (torch.from_numpy(canvas[..., ::-1].copy())
              .permute(2, 0, 1).unsqueeze(0)
              .float().to(device) / 255.0)
    return tensor, scale, left, top


def decode_p3(cv2_out, cv3_out, img_size, orig_w, orig_h,
              scale, pad_x, pad_y, conf_thresh, iou_thresh):
    stride = HEAD_STRIDE[0]
    device = cv2_out.device
    _, reg_ch, fh, fw = cv2_out.shape
    reg_max = reg_ch // 4

    ys = torch.arange(fh, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(fw, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    anchors   = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    n_anchors = fh * fw

    box_logits = cv2_out[0].permute(1, 2, 0).reshape(n_anchors, 4, reg_max)
    box_dists  = (box_logits.softmax(-1) *
                  torch.arange(reg_max, device=device, dtype=torch.float32)).sum(-1)

    lt, rb = box_dists[:, :2], box_dists[:, 2:]
    x1 = (anchors[:, 0] - lt[:, 0]) * stride
    y1 = (anchors[:, 1] - lt[:, 1]) * stride
    x2 = (anchors[:, 0] + rb[:, 0]) * stride
    y2 = (anchors[:, 1] + rb[:, 1]) * stride

    nc         = cv3_out.shape[1]
    cls_logits = cv3_out[0].permute(1, 2, 0).reshape(n_anchors, nc)
    conf, cls  = cls_logits.sigmoid().max(dim=-1)

    mask = conf > conf_thresh
    if mask.sum() == 0:
        return []

    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    conf, cls       = conf[mask], cls[mask]
    boxes_t = torch.stack([x1, y1, x2, y2], dim=1)
    keep    = tv_ops.batched_nms(boxes_t, conf, cls, iou_thresh)
    boxes_t, conf, cls = boxes_t[keep].cpu(), conf[keep].cpu(), cls[keep].cpu()

    out = []
    for (bx1, by1, bx2, by2), c, cl in zip(boxes_t.tolist(), conf.tolist(), cls.tolist()):
        ox1 = max(0,      (bx1 - pad_x) / scale)
        oy1 = max(0,      (by1 - pad_y) / scale)
        ox2 = min(orig_w, (bx2 - pad_x) / scale)
        oy2 = min(orig_h, (by2 - pad_y) / scale)
        if ox2 > ox1 and oy2 > oy1:
            out.append((ox1, oy1, ox2, oy2, float(c), int(cl)))
    return out


# ── Detection helpers ──────────────────────────────────────────────────────────

def size_bucket(xyxy, img_w, img_h):
    """0=small, 1=medium, 2=large — based on relative box side-length."""
    rel = ((xyxy[2] - xyxy[0]) / img_w * (xyxy[3] - xyxy[1]) / img_h) ** 0.5
    if rel < SIZE_THRESHS[0]:
        return 0
    elif rel < SIZE_THRESHS[1]:
        return 1
    return 2


def extract_full(results, img_w, img_h):
    """Return list of (x1,y1,x2,y2, conf, cls_id, size_bucket) from predict()."""
    out = []
    if not results or results[0].boxes is None:
        return out
    for box in results[0].boxes:
        xyxy = box.xyxy[0].tolist()
        out.append((*xyxy, float(box.conf[0]), int(box.cls[0]),
                    size_bucket(xyxy, img_w, img_h)))
    return out


def extract_p3(detections, img_w, img_h):
    """Return same tuple format from decode_p3() output."""
    return [(x1, y1, x2, y2, c, cl, size_bucket((x1,y1,x2,y2), img_w, img_h))
            for (x1, y1, x2, y2, c, cl) in detections]


def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_boxes(frame, dets, colour_map, label_prefix=""):
    for (x1, y1, x2, y2, conf, cls_id, bucket) in dets:
        color = colour_map[bucket]
        name  = CLASSES.get(cls_id, str(cls_id))
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, THICKNESS)
        label = f"{label_prefix}{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (ix1, iy1 - th - bl - 3), (ix1 + tw + 3, iy1), color, -1)
        cv2.putText(frame, label, (ix1 + 1, iy1 - bl - 1),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_panel_label(frame, text, colour):
    """Large semi-transparent label at top-centre of a panel."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    x = (frame.shape[1] - tw) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, 4), (x + tw + 6, th + 14), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x, th + 8), FONT, 0.7, colour, 2, cv2.LINE_AA)
    return frame


def draw_hud(canvas, frame_no, full_ms_frame, p3_ms_frame,
             full_dets, p3_dets, total_full_ms, total_p3_ms, w):
    """
    HUD bar at the bottom of the combined canvas.
    Shows per-frame timing and running totals for both paths.
    """
    bar_h = 56
    y0    = canvas.shape[0] - bar_h
    cv2.rectangle(canvas, (0, y0), (w, canvas.shape[0]), (12, 12, 18), -1)
    cv2.line(canvas, (0, y0), (w, y0), (55, 55, 70), 1)

    # Per-frame numbers
    speedup_frame = full_ms_frame / max(p3_ms_frame, 1e-3)
    speedup_total = total_full_ms / max(total_p3_ms, 1e-3)

    left_col = [
        f"Frame {frame_no}",
        f"Full: {full_ms_frame:.1f}ms  |  P3: {p3_ms_frame:.1f}ms  |  "
        f"Frame speedup: {speedup_frame:.2f}x",
    ]
    right_col = [
        f"Running total — Full: {total_full_ms/1000:.2f}s  P3: {total_p3_ms/1000:.2f}s  "
        f"Speedup: {speedup_total:.2f}x",
        f"Dets this frame — Full: {len(full_dets)}   P3: {len(p3_dets)}",
    ]

    for i, line in enumerate(left_col):
        cv2.putText(canvas, line, (10, y0 + 16 + i * 18),
                    FONT, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    for i, line in enumerate(right_col):
        cv2.putText(canvas, line, (w // 2 + 10, y0 + 16 + i * 18),
                    FONT, 0.46, (210, 210, 210), 1, cv2.LINE_AA)

    cv2.line(canvas, (w // 2, y0), (w // 2, canvas.shape[0]), (55, 55, 70), 1)
    return canvas


# ── Summary card ───────────────────────────────────────────────────────────────

def build_summary_card(h, w, total_frames, duration_s,
                       total_full_ms, total_p3_ms,
                       full_class_counts, p3_class_counts,
                       full_size_counts,  p3_size_counts,
                       model_name):
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (15, 15, 20)
    cw = w

    def put(text, x, y, scale=0.55, col=(220, 220, 220), bold=False):
        cv2.putText(card, text, (x, y), FONT, scale, col,
                    2 if bold else 1, cv2.LINE_AA)

    # Title
    title = "FULL MODEL vs P3-ONLY  —  COMPARISON SUMMARY"
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.85, 2)
    put(title, (cw - tw) // 2, 52, scale=0.85, col=(240, 240, 240), bold=True)
    cv2.line(card, (cw // 8, 66), (7 * cw // 8, 66), (55, 55, 68), 1)

    # Metadata strip
    speedup   = total_full_ms / max(total_p3_ms, 1e-3)
    saved_s   = (total_full_ms - total_p3_ms) / 1000
    meta = [
        ("Model",    model_name),
        ("Frames",   str(total_frames)),
        ("Duration", f"{duration_s:.1f}s"),
        ("Speedup",  f"{speedup:.2f}x"),
    ]
    col_w = cw // len(meta)
    for i, (lbl, val) in enumerate(meta):
        x = col_w * i + col_w // 2 - 30
        put(lbl, x, 90,  scale=0.45, col=(120, 120, 140))
        put(val, x, 108, scale=0.65, col=(230, 230, 230), bold=True)
    cv2.line(card, (cw // 8, 122), (7 * cw // 8, 122), (40, 40, 55), 1)

    # ── Timing block ──────────────────────────────────────────────────────────
    by0  = 138
    bx0, bx1 = int(cw * 0.04), int(cw * 0.96)
    cv2.rectangle(card, (bx0, by0), (bx1, by0 + 90), (24, 24, 38), -1)
    cv2.rectangle(card, (bx0, by0), (bx1, by0 + 90), (55, 55, 75),  1)
    put("TIMING", bx0 + 14, by0 + 22, scale=0.60, col=(180, 180, 210), bold=True)

    timing_rows = [
        ("Full model total / avg per frame",
         f"{total_full_ms/1000:.2f}s  |  {total_full_ms/max(total_frames,1):.2f} ms/frame",
         (190, 190, 190)),
        ("P3 path total / avg per frame",
         f"{total_p3_ms/1000:.2f}s  |  {total_p3_ms/max(total_frames,1):.2f} ms/frame",
         (100, 255, 160)),
        ("Time saved  |  Speedup",
         f"{saved_s:.2f}s  |  {speedup:.2f}x",
         (100, 200, 255)),
    ]
    for i, (lbl, val, col) in enumerate(timing_rows):
        put(f"{lbl} : {val}", bx0 + 14, by0 + 44 + i * 20, scale=0.48, col=col)

    # ── Detection counts ──────────────────────────────────────────────────────
    dy0 = by0 + 108
    cv2.line(card, (cw // 8, dy0 - 6), (7 * cw // 8, dy0 - 6), (40, 40, 55), 1)
    put("DETECTIONS  —  COUNT  BY  CLASS  AND  SIZE",
        cw // 2 - 210, dy0 + 14, scale=0.62, col=(180, 180, 200), bold=True)

    # Column headers
    col_xs = [int(cw * f) for f in (0.04, 0.28, 0.44, 0.60, 0.76)]
    headers = ["", "Full model", "P3 only", "Delta (F−P3)", "Delta %"]
    for xi, hdr in zip(col_xs, headers):
        put(hdr, xi, dy0 + 34, scale=0.44, col=(120, 120, 150))
    cv2.line(card, (bx0, dy0 + 40), (bx1, dy0 + 40), (35, 35, 50), 1)

    # Class rows
    all_cls = sorted(set(list(full_class_counts) + list(p3_class_counts)))
    row_y   = dy0 + 58
    for cls_id in all_cls:
        fc  = full_class_counts.get(cls_id, 0)
        pc  = p3_class_counts.get(cls_id, 0)
        d   = fc - pc
        pct = (d / max(fc, 1)) * 100
        col = (100, 255, 160) if d >= 0 else (255, 120, 80)
        put(CLASSES.get(cls_id, str(cls_id)), col_xs[0], row_y, scale=0.50)
        put(str(fc),              col_xs[1], row_y, scale=0.50)
        put(str(pc),              col_xs[2], row_y, scale=0.50)
        put(f"{d:+d}",            col_xs[3], row_y, scale=0.50, col=col)
        put(f"{pct:+.1f}%",       col_xs[4], row_y, scale=0.50, col=col)
        row_y += 20

    # Size rows
    row_y += 8
    cv2.line(card, (bx0, row_y - 4), (bx1, row_y - 4), (35, 35, 50), 1)
    for bucket in range(3):
        fs  = full_size_counts.get(bucket, 0)
        ps  = p3_size_counts.get(bucket, 0)
        d   = fs - ps
        pct = (d / max(fs, 1)) * 100
        col = (100, 255, 160) if d >= 0 else (255, 120, 80)
        put(SIZE_LABELS[bucket], col_xs[0], row_y, scale=0.50, col=(160, 160, 180))
        put(str(fs),             col_xs[1], row_y, scale=0.50)
        put(str(ps),             col_xs[2], row_y, scale=0.50)
        put(f"{d:+d}",           col_xs[3], row_y, scale=0.50, col=col)
        put(f"{pct:+.1f}%",      col_xs[4], row_y, scale=0.50, col=col)
        row_y += 20

    # Footer note
    note = ("Both paths ran on every frame. P3 uses the early-exit DFL decoder. "
            "Size buckets: small <15%, medium 15-40%, large >40% of frame diagonal.")
    (fw2, _), _ = cv2.getTextSize(note, FONT, 0.34, 1)
    put(note, (cw - fw2) // 2, h - 18, scale=0.34, col=(65, 65, 82))

    return card


# ── CSV writers ────────────────────────────────────────────────────────────────

def open_det_csv(path, label):
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["frame", "path", "x1", "y1", "x2", "y2",
                "conf", "cls_id", "cls_name", "size_bucket", "size_label"])
    return f, w


def write_det_rows(writer, frame_no, path_label, dets):
    for (x1, y1, x2, y2, conf, cls_id, bucket) in dets:
        writer.writerow([
            frame_no, path_label,
            f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
            f"{conf:.4f}", cls_id, CLASSES.get(cls_id, str(cls_id)),
            bucket, SIZE_LABELS[bucket],
        ])


# ── Main ───────────────────────────────────────────────────────────────────────

def run(weights, source,
        output=None, out_dir=None,
        conf=0.25, iou=0.45,
        img_size=640, summary_frames=90,
        yaml_cfg="yolov8n.yaml"):

    # 1. Load model + build P3 module
    print(f"\n[INFO] Loading model : {weights}")
    model  = load_model(weights, yaml_cfg)
    layers = list(model.model.model)
    ee_p3  = EarlyExitModel(layers, trace_head_path(layers, 0), 0).to(DEVICE).eval()
    print("[INFO] P3 early-exit module ready")

    # 2. Video metadata
    cap_probe = cv2.VideoCapture(source)
    if not cap_probe.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    src_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    print(f"[INFO] Video  : {src_w}x{src_h}  {src_fps:.1f}fps  ~{total_f} frames")
    print(f"[INFO] Device : {DEVICE}  conf={conf}  iou={iou}")

    # 3. Output paths
    src_path = Path(source)
    out_dir  = Path(out_dir) if out_dir else src_path.parent / "compare_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if output is None:
        output = str(out_dir / (src_path.stem + "_compare.mp4"))

    hud_h   = 56                          # HUD bar height in pixels
    out_w   = src_w * 2                   # side-by-side
    out_h   = src_h + hud_h
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(output, fourcc, src_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output}")

    timing_csv_path = out_dir / "per_frame_timing.csv"
    full_csv_path   = out_dir / "full_detections.csv"
    p3_csv_path     = out_dir / "p3_detections.csv"

    timing_f  = open(timing_csv_path, "w", newline="")
    timing_w  = csv.writer(timing_f)
    timing_w.writerow(["frame", "full_ms", "p3_ms", "speedup_frame",
                        "full_det_count", "p3_det_count"])

    full_det_f, full_det_w = open_det_csv(full_csv_path, "full")
    p3_det_f,   p3_det_w   = open_det_csv(p3_csv_path,   "p3")

    print(f"[INFO] Output video : {output}")
    print(f"[INFO] CSVs         : {out_dir}\n")

    # 4. Accumulators
    total_full_ms   = 0.0
    total_p3_ms     = 0.0
    full_class_counts = {}
    p3_class_counts   = {}
    full_size_counts  = {}
    p3_size_counts    = {}
    frame_no          = 0
    t_wall            = time.perf_counter()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot re-open: {source}")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_no += 1

        # ── A. Full model ──────────────────────────────────────────────────────
        gpu_sync()
        t0 = time.perf_counter()
        results = model.predict(
            bgr, imgsz=img_size, conf=conf, iou=iou,
            device=DEVICE, verbose=False, classes=list(CLASSES.keys()),
        )
        gpu_sync()
        full_ms = (time.perf_counter() - t0) * 1000
        total_full_ms += full_ms

        full_dets = extract_full(results, src_w, src_h)

        # ── B. P3 early-exit ───────────────────────────────────────────────────
        gpu_sync()
        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess_frame(bgr, img_size, DEVICE)
        with torch.no_grad():
            cv2_out, cv3_out = ee_p3(tensor)
        raw_p3    = decode_p3(cv2_out, cv3_out, img_size, src_w, src_h,
                              scale, pad_x, pad_y, conf, iou)
        gpu_sync()
        p3_ms = (time.perf_counter() - t0) * 1000
        total_p3_ms += p3_ms

        p3_dets = extract_p3(raw_p3, src_w, src_h)

        # ── C. Accumulate stats ────────────────────────────────────────────────
        for (_, _, _, _, _, cls_id, bucket) in full_dets:
            full_class_counts[cls_id]  = full_class_counts.get(cls_id, 0)  + 1
            full_size_counts[bucket]   = full_size_counts.get(bucket, 0)   + 1

        for (_, _, _, _, _, cls_id, bucket) in p3_dets:
            p3_class_counts[cls_id]    = p3_class_counts.get(cls_id, 0)    + 1
            p3_size_counts[bucket]     = p3_size_counts.get(bucket, 0)     + 1

        # ── D. Write CSVs ──────────────────────────────────────────────────────
        spd_frame = full_ms / max(p3_ms, 1e-3)
        timing_w.writerow([frame_no, f"{full_ms:.4f}", f"{p3_ms:.4f}",
                            f"{spd_frame:.4f}",
                            len(full_dets), len(p3_dets)])
        write_det_rows(full_det_w, frame_no, "full", full_dets)
        write_det_rows(p3_det_w,   frame_no, "p3",   p3_dets)

        # ── E. Draw side-by-side frame ─────────────────────────────────────────
        left_panel  = bgr.copy()
        right_panel = bgr.copy()

        draw_boxes(left_panel,  full_dets, COLOUR_FULL)
        draw_boxes(right_panel, p3_dets,   COLOUR_P3)
        draw_panel_label(left_panel,  "FULL MODEL", (80, 200, 255))
        draw_panel_label(right_panel, "P3 ONLY",    (255, 140, 50))

        canvas       = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:src_h, :src_w]      = left_panel
        canvas[:src_h, src_w:out_w] = right_panel
        cv2.line(canvas, (src_w, 0), (src_w, src_h), (80, 80, 100), 2)

        draw_hud(canvas, frame_no, full_ms, p3_ms,
                 full_dets, p3_dets, total_full_ms, total_p3_ms, out_w)
        writer.write(canvas)

        # ── F. Progress print ──────────────────────────────────────────────────
        if frame_no % 50 == 0 or frame_no == 1:
            pct = frame_no / max(total_f, 1) * 100
            spd = total_full_ms / max(total_p3_ms, 1e-3)
            print(f"  [{frame_no:>6}/{total_f}]  {pct:5.1f}%  "
                  f"full={full_ms:.1f}ms  p3={p3_ms:.1f}ms  "
                  f"running_speedup={spd:.2f}x  "
                  f"full_dets={len(full_dets)}  p3_dets={len(p3_dets)}")

    duration_s = frame_no / src_fps
    cap.release()
    timing_f.close()
    full_det_f.close()
    p3_det_f.close()

    # 5. Summary card
    print(f"\n[INFO] Appending summary card ({summary_frames} frames) ...")
    card_full = build_summary_card(
        out_h, out_w, frame_no, duration_s,
        total_full_ms, total_p3_ms,
        full_class_counts, p3_class_counts,
        full_size_counts,  p3_size_counts,
        Path(weights).stem,
    )
    for _ in range(summary_frames):
        writer.write(card_full)
    writer.release()

    # 6. Summary CSV
    summary_path = out_dir / "summary.csv"
    speedup      = total_full_ms / max(total_p3_ms, 1e-3)
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow(["metric", "full_model", "p3_only", "delta", "delta_pct"])

        # Timing
        sw.writerow(["total_inference_ms",
                     f"{total_full_ms:.2f}", f"{total_p3_ms:.2f}",
                     f"{total_full_ms - total_p3_ms:.2f}",
                     f"{(total_full_ms - total_p3_ms) / max(total_full_ms,1)*100:.2f}"])
        sw.writerow(["avg_ms_per_frame",
                     f"{total_full_ms/max(frame_no,1):.4f}",
                     f"{total_p3_ms/max(frame_no,1):.4f}",
                     "", f"speedup={speedup:.4f}x"])

        # Total detections
        total_full = sum(full_class_counts.values())
        total_p3   = sum(p3_class_counts.values())
        sw.writerow(["total_detections",
                     total_full, total_p3,
                     total_full - total_p3,
                     f"{(total_full - total_p3)/max(total_full,1)*100:.2f}"])

        # Per-class
        all_cls = sorted(set(list(full_class_counts) + list(p3_class_counts)))
        for cls_id in all_cls:
            fc = full_class_counts.get(cls_id, 0)
            pc = p3_class_counts.get(cls_id, 0)
            sw.writerow([f"class_{CLASSES.get(cls_id, cls_id)}",
                         fc, pc, fc - pc,
                         f"{(fc-pc)/max(fc,1)*100:.2f}"])

        # Per-size
        for bucket in range(3):
            fs = full_size_counts.get(bucket, 0)
            ps = p3_size_counts.get(bucket, 0)
            sw.writerow([f"size_{SIZE_LABELS[bucket]}",
                         fs, ps, fs - ps,
                         f"{(fs-ps)/max(fs,1)*100:.2f}"])

    # 7. Terminal report
    wall = time.perf_counter() - t_wall
    print(f"\n{'='*65}")
    print(f"  COMPARISON REPORT")
    print(f"{'='*65}")
    print(f"  Frames processed     : {frame_no}")
    print(f"  Duration             : {duration_s:.1f}s")
    print(f"  Wall time (incl I/O) : {wall:.1f}s")
    print()
    print(f"  -- Timing ------------------------------------------")
    print(f"  Full model total     : {total_full_ms/1000:.2f}s  "
          f"[{total_full_ms/max(frame_no,1):.2f} ms/frame]")
    print(f"  P3 path total        : {total_p3_ms/1000:.2f}s  "
          f"[{total_p3_ms/max(frame_no,1):.2f} ms/frame]")
    print(f"  Time saved           : {(total_full_ms-total_p3_ms)/1000:.2f}s")
    print(f"  Speedup              : {speedup:.2f}x")
    print()
    print(f"  -- Detections by class -----------------------------")
    print(f"  {'Class':<12}  {'Full':>8}  {'P3':>8}  {'Delta':>8}  {'Delta%':>8}")
    all_cls = sorted(set(list(full_class_counts) + list(p3_class_counts)))
    for cls_id in all_cls:
        fc  = full_class_counts.get(cls_id, 0)
        pc  = p3_class_counts.get(cls_id, 0)
        d   = fc - pc
        pct = (d / max(fc, 1)) * 100
        print(f"  {CLASSES.get(cls_id, str(cls_id)):<12}  {fc:>8}  {pc:>8}  "
              f"{d:>+8}  {pct:>+7.1f}%")
    print()
    print(f"  -- Detections by size ------------------------------")
    print(f"  {'Size':<12}  {'Full':>8}  {'P3':>8}  {'Delta':>8}  {'Delta%':>8}")
    for bucket in range(3):
        fs  = full_size_counts.get(bucket, 0)
        ps  = p3_size_counts.get(bucket, 0)
        d   = fs - ps
        pct = (d / max(fs, 1)) * 100
        print(f"  {SIZE_LABELS[bucket]:<12}  {fs:>8}  {ps:>8}  "
              f"{d:>+8}  {pct:>+7.1f}%")
    print()
    print(f"  -- Output files ------------------------------------")
    print(f"  Video      : {output}")
    print(f"  Timing CSV : {timing_csv_path}")
    print(f"  Full dets  : {full_csv_path}")
    print(f"  P3 dets    : {p3_csv_path}")
    print(f"  Summary    : {summary_path}")
    print(f"{'='*65}\n")

    return output


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="YOLOv8 full-model vs P3-only comparison — both paths on every frame"
    )
    p.add_argument("--weights",        default="C:/Users/gr3g2/Downloads/nano_lr.pt")
    p.add_argument("--source",         default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output.mp4")
    p.add_argument("--output",         default="C:/Users/gr3g2/Desktop/MUSIT_MDM/sailboat/output_b.mp4",
                   help="Output video path (default: <out_dir>/<stem>_compare.mp4)")
    p.add_argument("--out-dir",        default=None,
                   help="Directory for all outputs (default: <source_dir>/compare_output)")
    p.add_argument("--conf",           type=float, default=0.25)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--img-size",       type=int,   default=640)
    p.add_argument("--summary-frames", type=int,   default=90)
    p.add_argument("--yaml",           default="yolov8n.yaml",
                   help="Architecture YAML — must match checkpoint")
    args = p.parse_args()

    run(
        weights        = args.weights,
        source         = args.source,
        output         = args.output,
        out_dir        = args.out_dir,
        conf           = args.conf,
        iou            = args.iou,
        img_size       = args.img_size,
        summary_frames = args.summary_frames,
        yaml_cfg       = args.yaml,
    )