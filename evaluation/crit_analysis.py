"""
infer_video.py — YOLOv8 video inference with proximity-signal head routing,
                 per-head attribution, rigorous speedup measurement, and
                 head-distribution comparison between always-full and
                 signal-routed strategies.

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

Head-distribution comparison:
  • On every P3-signal frame the full model's raw head outputs are also
    computed silently (no timing, results discarded after attribution) to
    build baseline_head_counts — the detection distribution that always-full
    inference would have produced.
  • CONSISTENCY FIX: The baseline P3 count on P3-signal frames is obtained
    by running decode_p3 on the raw P3 cv2/cv3 outputs from FullRawModel —
    the EXACT same decode path used by EarlyExitModel. This guarantees that
    the P3 detection count is identical between routed and baseline strategies
    for the same frame, isolating the cost of skipping P4/P5.
  • P4/P5 baseline counts come from size-heuristic attribution on the full
    model's ultralytics-decoded detections (unchanged from original).
  • routed_head_counts tracks what the signal-routed path actually produced.
  • Both distributions are shown side-by-side in the summary card and
    terminal report.

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

# ── Profiling constants ────────────────────────────────────────────────────────
N_WARMUP = 20
N_TIMING = 100

DEFAULT_DIST_THRESHOLD = 20.0
DEFAULT_RATE_THRESHOLD = 0.5

_R_EARTH = 6_371_000.0


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE OBJECT TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class SimpleTracker:
    """
    Basic centroid tracking for P3 detections.
    Assigns track IDs based on closest centroid matching.
    """
    def __init__(self, max_distance=50):
        self.next_id = 1
        self.tracks = {}  # id -> (cx, cy, cls, frames_missing)
        self.max_distance = max_distance
        self.max_missing = 5

    def update(self, detections):
        """
        Update tracking with new detections.
        Returns: list of (x1, y1, x2, y2, conf, cls, track_id)
        """
        import numpy as np

        if not detections:
            # Age out missing tracks
            self.tracks = {tid: (cx, cy, cls, miss+1) 
                          for tid, (cx, cy, cls, miss) in self.tracks.items()
                          if miss+1 <= self.max_missing}
            return []

        # Calculate centroids for new detections
        new_centroids = []
        for x1, y1, x2, y2, conf, cls in detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            new_centroids.append((cx, cy, cls))

        tracked_detections = []
        used_track_ids = set()

        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            cx, cy, det_cls = new_centroids[i]
            best_id = None
            min_dist = float('inf')

            # Find closest existing track of same class
            for tid, (tx, ty, tcls, miss) in self.tracks.items():
                if tcls != det_cls or tid in used_track_ids:
                    continue

                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = tid

            if best_id is not None:
                # Update existing track
                self.tracks[best_id] = (cx, cy, det_cls, 0)
                used_track_ids.add(best_id)
                track_id = best_id
            else:
                # Create new track
                track_id = self.next_id
                self.tracks[track_id] = (cx, cy, det_cls, 0)
                self.next_id += 1
                used_track_ids.add(track_id)

            tracked_detections.append((x1, y1, x2, y2, conf, cls, track_id))

        # Age out unused tracks
        for tid in list(self.tracks.keys()):
            if tid not in used_track_ids:
                cx, cy, cls, miss = self.tracks[tid]
                if miss + 1 > self.max_missing:
                    del self.tracks[tid]
                else:
                    self.tracks[tid] = (cx, cy, cls, miss + 1)

        return tracked_detections


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

    # Per-second data for CSV export
    per_second_data = []
    for i, ts in enumerate(grid):
        per_second_data.append({
            "timestamp": str(ts),
            "timestamp_unix": ts.timestamp(),
            "signal": per_sec[i], 
            "distance_m": round(float(dists[i]), 2),
            "closure_rate_ms": round(float(cr[i]), 3),
            "mmsi_a": mmsi_a,
            "mmsi_b": mmsi_b,
        })

    print(f"[DEBUG] compute_proximity_signals returning {len(per_second_data)} per-second records")

    return frame_signals, summary, per_second_data

#  MODEL LOADING

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

#  GRAPH TRACING & EARLY-EXIT MODULES

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
    No postprocessing / NMS.  Used for latency profiling only.
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

class FullRawModel(nn.Module):
    """
    Full backbone + neck + ALL three head branches, returning raw (cv2, cv3)
    tensors per head without any decode or NMS.

    This is the backbone of the consistency fix: on P3-signal frames we run
    this model silently and then apply decode_p3 on head[0]'s raw outputs,
    guaranteeing that the baseline P3 count uses the IDENTICAL decode path as
    the EarlyExitModel.  P4/P5 baseline counts are derived separately via the
    size-heuristic attribution on the full ultralytics-decoded results.

    Returns:
        List of (cv2_out, cv3_out) tensors, one tuple per head (P3, P4, P5).
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

        raw_heads = []
        for i, fi in enumerate(det_f):
            feat = cache[fi]
            raw_heads.append((self.cv2[i](feat), self.cv3[i](feat)))
        # raw_heads[0] → P3, raw_heads[1] → P4, raw_heads[2] → P5
        return raw_heads


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
#  RIGOROUS LATENCY PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def measure_time(module, dummy, n_warmup=N_WARMUP, n_runs=N_TIMING, label=""):
    """
    Warmup → synchronise → time N_TIMING individual forward passes →
    return the MEDIAN latency in milliseconds.
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
    Measure FullForwardModel and P3 EarlyExitModel.
    Returns: full_ms, p3_ms, speedup
    """
    layers = list(model.model.model)
    dummy  = torch.zeros(1, 3, img_size, img_size, device=DEVICE)

    full_mod = FullForwardModel(layers).to(DEVICE)
    print(f"\n[PROFILE] Full forward model ({N_WARMUP} warmup, {N_TIMING} runs) …")
    full_ms = measure_time(full_mod, dummy, label="full")
    print(f"  → median {full_ms:.3f} ms/frame")
    del full_mod

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
    print(f"  Full model   : {full_ms:.3f} ms/frame")
    print(f"  P3 early-exit: {p3_ms:.3f} ms/frame")
    print(f"  Speedup      : {speedup:.2f}×")
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

    tensor = (torch.from_numpy(canvas)
              .permute(2, 0, 1)
              .unsqueeze(0)
              .float()
              .to(device) / 255.0)
    return tensor, scale, left, top


def _decode_head_generic(cv2_out, cv3_out, head_idx, img_size, orig_w, orig_h,
                        scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    Generic decode function for any head (P3, P4, P5).
    All heads use the same decode logic, only stride differs.
    """
    stride = HEAD_STRIDE[head_idx]
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
    y2 = (anchors[:, 1] - rb[:, 1]) * stride

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


def decode_p3(cv2_out, cv3_out, img_size, orig_w, orig_h,
              scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    Decode raw P3 head outputs into (x1,y1,x2,y2,conf,cls) detections.
    This is the canonical P3 decode path used by both EarlyExitModel and FullRawModel.
    """
    return _decode_head_generic(cv2_out, cv3_out, 0, img_size, orig_w, orig_h,
                               scale, pad_x, pad_y, conf_thresh, iou_thresh)


def decode_p4(cv2_out, cv3_out, img_size, orig_w, orig_h,
              scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    Decode raw P4 head outputs into (x1,y1,x2,y2,conf,cls) detections.
    Uses identical logic to decode_p3, only stride differs (16 vs 8).
    """
    return _decode_head_generic(cv2_out, cv3_out, 1, img_size, orig_w, orig_h,
                               scale, pad_x, pad_y, conf_thresh, iou_thresh)


def decode_p5(cv2_out, cv3_out, img_size, orig_w, orig_h,
              scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    Decode raw P5 head outputs into (x1,y1,x2,y2,conf,cls) detections.
    Uses identical logic to decode_p3, only stride differs (32 vs 8).
    """
    return _decode_head_generic(cv2_out, cv3_out, 2, img_size, orig_w, orig_h,
                               scale, pad_x, pad_y, conf_thresh, iou_thresh)


def decode_all_heads(raw_heads, img_size, orig_w, orig_h,
                    scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    Decode all three heads (P3, P4, P5) from FullRawModel outputs.
    Returns detections grouped by head: {0: p3_dets, 1: p4_dets, 2: p5_dets}

    Args:
        raw_heads: List of (cv2_out, cv3_out) tuples from FullRawModel
    """
    decode_funcs = [decode_p3, decode_p4, decode_p5]
    head_detections = {}

    for head_idx, (cv2_out, cv3_out) in enumerate(raw_heads):
        detections = decode_funcs[head_idx](
            cv2_out, cv3_out, img_size, orig_w, orig_h,
            scale, pad_x, pad_y, conf_thresh, iou_thresh
        )
        head_detections[head_idx] = detections

    return head_detections

#  DETECTION ATTRIBUTION & DRAWING

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
    frame_class_counts = {cls: 0 for cls in CLASSES.keys()}

    if results is None or len(results) == 0:
        return frame, frame_head_counts, frame_class_counts
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame, frame_head_counts, frame_class_counts

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
        frame_class_counts[cls] += 1

    return frame, frame_head_counts, frame_class_counts


def draw_detections_from_list(frame, detections, img_w, img_h, show_tracking=False):
    frame_head_counts = {0: 0, 1: 0, 2: 0}
    frame_class_counts = {cls: 0 for cls in CLASSES.keys()}

    if not detections:
        return frame, frame_head_counts, frame_class_counts

    for detection in detections:
        if show_tracking and len(detection) >= 7:
            x1, y1, x2, y2, conf, cls, track_id = detection
        else:
            x1, y1, x2, y2, conf, cls = detection[:6]
            track_id = None

        head  = 0   # P3 early-exit: all detections are from P3 head by definition
        color = HEAD_COLOUR[head]
        name  = CLASSES.get(cls, str(cls))
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, THICKNESS)

        if show_tracking and track_id is not None:
            label = f"{HEAD_ABBREV[head]} {name} #{track_id} {conf:.2f}"
        else:
            label = f"{HEAD_ABBREV[head]} {name} {conf:.2f}"

        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (ix1, iy1 - th - bl - 4), (ix1 + tw + 4, iy1), color, -1)
        cv2.putText(frame, label, (ix1 + 2, iy1 - bl - 2),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

        frame_head_counts[head] += 1
        frame_class_counts[cls] += 1

    return frame, frame_head_counts, frame_class_counts


def draw_detections_from_heads(frame, head_detections, img_w, img_h):
    """
    Draw detections from decode_all_heads output where detections are already
    properly attributed by head. This ensures consistent attribution between
    FULL and P3-only paths.

    Args:
        head_detections: {0: p3_dets, 1: p4_dets, 2: p5_dets} from decode_all_heads
    """
    frame_head_counts = {0: 0, 1: 0, 2: 0}
    frame_class_counts = {cls: 0 for cls in CLASSES.keys()}

    for head_idx, detections in head_detections.items():
        if not detections:
            continue

        color = HEAD_COLOUR[head_idx]

        for x1, y1, x2, y2, conf, cls in detections:
            name = CLASSES.get(cls, str(cls))
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, THICKNESS)
            label = f"{HEAD_ABBREV[head_idx]} {name} {conf:.2f}"

            (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            cv2.rectangle(frame, (ix1, iy1 - th - bl - 4), (ix1 + tw + 4, iy1), color, -1)
            cv2.putText(frame, label, (ix1 + 2, iy1 - bl - 2),
                        FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

            frame_head_counts[head_idx] += 1
            frame_class_counts[cls] += 1

    return frame, frame_head_counts, frame_class_counts


def draw_hud(frame, frame_no, total_head_counts, fps_live,
             active_head, signal, speedup,
             p3_class_counts=None, active_tracks=None):
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

    # Add P3 class breakdown if available
    if signal == "P3" and p3_class_counts:
        lines.append("")
        lines.append("P3 Classes:")
        for cls, name in CLASSES.items():
            count = p3_class_counts.get(cls, 0)
            lines.append(f"  {name}: {count}")

        # Add active tracks info
        if active_tracks:
            lines.append("")
            lines.append(f"Active tracks: {len(active_tracks)}")
            # Show track IDs (limit to first 5)
            track_ids = sorted(active_tracks.keys())[:5]
            if track_ids:
                ids_str = ", ".join(f"#{tid}" for tid in track_ids)
                if len(active_tracks) > 5:
                    ids_str += f" (+{len(active_tracks)-5})"
                lines.append(f"IDs: {ids_str}")

    panel_w = 270  # Slightly wider for new content
    panel_h = 22 + len(lines) * 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        if line.startswith("Signal"):
            col = (50, 140, 255) if signal == "FULL" else (80, 220, 80)
        elif line.startswith("Speedup"):
            col = (100, 200, 255)
        elif line.startswith("P3 Classes:") or line.startswith("Active tracks:") or line.startswith("IDs:"):
            col = (255, 200, 100)  # Orange for P3 specific info
        elif (line.startswith("  ") and
              any(name in line for name in CLASSES.values())):
            col = (180, 255, 180)  # Light green for class counts
        elif (line.startswith("  ") and
              line.strip().startswith(HEAD_ABBREV[active_head])):
            col = HEAD_COLOUR[active_head]
        else:
            col = (220, 220, 220)
        cv2.putText(frame, line, (14, 26 + i * 18),
                    FONT, 0.50, col, 1, cv2.LINE_AA)
    return frame

#  SUMMARY CARD

def build_summary_card(h, w,
                       routed_head_counts, baseline_head_counts,
                       total_frames, duration_s,
                       full_ms, p3_ms, speedup,
                       n_full_frames, n_p3_frames,
                       model_name, prox_summary):
    card_h = h + 220
    card   = np.zeros((card_h, w, 3), dtype=np.uint8)
    card[:] = (15, 15, 20)
    cw = w

    routed_total   = sum(routed_head_counts.values())
    baseline_total = sum(baseline_head_counts.values())

    title = "INFERENCE SUMMARY"
    (tw, _), _ = cv2.getTextSize(title, FONT, 1.3, 2)
    cv2.putText(card, title, ((cw - tw) // 2, 65),
                FONT, 1.3, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.line(card, (cw // 8, 82), (7 * cw // 8, 82), (55, 55, 68), 1)

    meta_items = [
        ("Model",         model_name),
        ("Frames",        str(total_frames)),
        ("Duration",      f"{duration_s:.1f}s"),
        ("Routed dets",   str(routed_total)),
        ("Baseline dets", str(baseline_total)),
    ]
    col_w = cw // len(meta_items)
    for i, (label, val) in enumerate(meta_items):
        x = col_w * i + col_w // 2
        cv2.putText(card, label, (x - 38, 114), FONT, 0.44, (120, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(card, val,   (x - 38, 132), FONT, 0.62, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.line(card, (cw // 8, 148), (7 * cw // 8, 148), (40, 40, 55), 1)

    # ── Proximity routing stats ────────────────────────────────────────────────
    sig_y = 170
    cv2.putText(card, "PROXIMITY  SIGNAL  ROUTING", (cw // 2 - 155, sig_y),
                FONT, 0.70, (180, 180, 200), 1, cv2.LINE_AA)
    prox_items = [
        (f"Min dist: {prox_summary['min_distance_m']:.1f} m",
         (100, 255, 160)),
        (f"Max close rate: {abs(prox_summary['max_closing_rate_ms']):.2f} m/s",
         (100, 200, 255)),
        (f"FULL frames: {n_full_frames}",
         (50,  140, 255)),
        (f"P3-only frames: {n_p3_frames}",
         (80,  220,  80)),
    ]
    px = int(cw * 0.06)
    for i, (txt, col) in enumerate(prox_items):
        cv2.putText(card, txt,
                    (px + (i % 2) * (cw // 2), sig_y + 22 + (i // 2) * 20),
                    FONT, 0.52, col, 1, cv2.LINE_AA)
    cv2.line(card, (cw // 8, sig_y + 52), (7 * cw // 8, sig_y + 52), (40, 40, 55), 1)

    # ── Per-head bars — signal-routed ──────────────────────────────────────────
    bar_y0    = sig_y + 75
    bar_x0    = int(cw * 0.22)
    bar_max_w = int(cw * 0.52)
    row_h     = 52
    cv2.putText(card, "DETECTIONS  BY  HEAD  —  SIGNAL-ROUTED  PATH",
                (cw // 2 - 225, bar_y0 - 8),
                FONT, 0.70, (180, 180, 200), 1, cv2.LINE_AA)

    for idx in range(3):
        count = routed_head_counts.get(idx, 0)
        color = HEAD_COLOUR[idx]
        y     = bar_y0 + 14 + idx * row_h
        pct   = count / max(routed_total, 1) * 100
        fill  = int(bar_max_w * count / max(routed_total, 1))
        cv2.putText(card, HEAD_NAMES[idx], (18, y + 22),
                    FONT, 0.62, color, 2, cv2.LINE_AA)
        cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + bar_max_w, y + 28), (38, 38, 50), -1)
        if fill > 0:
            cv2.rectangle(card, (bar_x0, y + 2), (bar_x0 + fill, y + 28), color, -1)
        cv2.putText(card, f"{count}  ({pct:.1f}%)",
                    (bar_x0 + bar_max_w + 12, y + 22),
                    FONT, 0.58, (210, 210, 210), 1, cv2.LINE_AA)
        lat = full_ms if idx > 0 else p3_ms
        cv2.putText(card, f"profiled latency: {lat:.2f} ms/frame",
                    (bar_x0, y + 42), FONT, 0.40, (100, 100, 118), 1, cv2.LINE_AA)

    # ── Rigorous latency comparison box ───────────────────────────────────────
    box_y  = bar_y0 + 14 + 3 * row_h + 18
    box_h2 = 148
    bx0    = int(cw * 0.06)
    bx1    = int(cw * 0.94)
    cv2.rectangle(card, (bx0, box_y), (bx1, box_y + box_h2), (28, 28, 42), -1)
    cv2.rectangle(card, (bx0, box_y), (bx1, box_y + box_h2), (60, 60, 82),  1)
    cv2.putText(card, "RIGOROUS  LATENCY  COMPARISON",
                (bx0 + 18, box_y + 26),
                FONT, 0.68, (200, 200, 220), 1, cv2.LINE_AA)

    proj_full_s  = full_ms * total_frames / 1000
    proj_mixed_s = (full_ms * n_full_frames + p3_ms * n_p3_frames) / 1000
    saved_s      = proj_full_s - proj_mixed_s

    timing_rows = [
        ("Profiling method",
         f"{N_WARMUP} warmup + {N_TIMING} runs, cuda.sync, median, no NMS",
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
                    (bx0 + 18, box_y + 46 + i * 20),
                    FONT, 0.42, col, 1, cv2.LINE_AA)

    # ── Head distribution comparison: always-full vs signal-routed ─────────────
    cmp_y   = box_y + box_h2 + 22
    cmp_h   = 185
    cmp_bx0 = bx0
    cmp_bx1 = bx1
    cv2.rectangle(card, (cmp_bx0, cmp_y), (cmp_bx1, cmp_y + cmp_h), (20, 28, 38), -1)
    cv2.rectangle(card, (cmp_bx0, cmp_y), (cmp_bx1, cmp_y + cmp_h), (55, 75, 95),  1)

    cv2.putText(card,
                "HEAD  DISTRIBUTION  COMPARISON  —  ALWAYS-FULL  vs  SIGNAL-ROUTED",
                (cmp_bx0 + 14, cmp_y + 24),
                FONT, 0.60, (200, 210, 230), 1, cv2.LINE_AA)
    cv2.putText(card,
                "(P3 baseline uses identical decode_p3 path as EarlyExitModel — "
                "counts are directly comparable)",
                (cmp_bx0 + 14, cmp_y + 40),
                FONT, 0.37, (100, 110, 130), 1, cv2.LINE_AA)

    col_bl_x  = cmp_bx0 + 200
    col_ro_x  = cmp_bx0 + 400
    col_dt_x  = cmp_bx0 + 590
    mini_bar  = 75
    hdr_y     = cmp_y + 58

    cv2.putText(card, "Head",          (cmp_bx0 + 14, hdr_y), FONT, 0.46, (155,165,185), 1, cv2.LINE_AA)
    cv2.putText(card, "Always-full",   (col_bl_x,     hdr_y), FONT, 0.46, (155,165,185), 1, cv2.LINE_AA)
    cv2.putText(card, "Signal-routed", (col_ro_x,     hdr_y), FONT, 0.46, (155,165,185), 1, cv2.LINE_AA)
    cv2.putText(card, "\u0394 dets",   (col_dt_x,     hdr_y), FONT, 0.46, (155,165,185), 1, cv2.LINE_AA)
    cv2.line(card, (cmp_bx0 + 10, hdr_y + 6),
             (cmp_bx1 - 10, hdr_y + 6), (50, 65, 80), 1)

    for idx in range(3):
        color = HEAD_COLOUR[idx]
        b_cnt = baseline_head_counts.get(idx, 0)
        r_cnt = routed_head_counts.get(idx, 0)
        b_pct = b_cnt / max(baseline_total, 1) * 100
        r_pct = r_cnt / max(routed_total,   1) * 100
        delta = r_cnt - b_cnt
        ry    = hdr_y + 22 + idx * 34

        cv2.putText(card, HEAD_NAMES[idx], (cmp_bx0 + 14, ry + 14),
                    FONT, 0.50, color, 1, cv2.LINE_AA)

        b_fill = int(mini_bar * b_cnt / max(baseline_total, 1))
        cv2.rectangle(card, (col_bl_x, ry), (col_bl_x + mini_bar, ry + 16), (38, 38, 50), -1)
        if b_fill > 0:
            cv2.rectangle(card, (col_bl_x, ry), (col_bl_x + b_fill, ry + 16), color, -1)
        cv2.putText(card, f"{b_cnt} ({b_pct:.1f}%)",
                    (col_bl_x + mini_bar + 5, ry + 13),
                    FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        r_fill = int(mini_bar * r_cnt / max(routed_total, 1))
        cv2.rectangle(card, (col_ro_x, ry), (col_ro_x + mini_bar, ry + 16), (38, 38, 50), -1)
        if r_fill > 0:
            cv2.rectangle(card, (col_ro_x, ry), (col_ro_x + r_fill, ry + 16), color, -1)
        cv2.putText(card, f"{r_cnt} ({r_pct:.1f}%)",
                    (col_ro_x + mini_bar + 5, ry + 13),
                    FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        delta_col = (80, 220, 80) if delta >= 0 else (80, 80, 220)
        cv2.putText(card, f"{delta:+d}", (col_dt_x, ry + 13),
                    FONT, 0.50, delta_col, 1, cv2.LINE_AA)

    tot_y   = hdr_y + 22 + 3 * 34
    t_delta = routed_total - baseline_total
    cv2.line(card, (cmp_bx0 + 10, tot_y - 4),
             (cmp_bx1 - 10, tot_y - 4), (50, 65, 80), 1)
    cv2.putText(card, "Total", (cmp_bx0 + 14, tot_y + 13),
                FONT, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(card, f"{baseline_total} (100%)",
                (col_bl_x + mini_bar + 5, tot_y + 13),
                FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(card, f"{routed_total} (100%)",
                (col_ro_x + mini_bar + 5, tot_y + 13),
                FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    td_col = (80, 220, 80) if t_delta >= 0 else (80, 80, 220)
    cv2.putText(card, f"{t_delta:+d}", (col_dt_x, tot_y + 13),
                FONT, 0.50, td_col, 1, cv2.LINE_AA)

    note = ("P3 counts: both strategies use decode_p3 on identical raw head outputs — "
            "Δ P3 will be 0 on P3-only frames; Δ P4/P5 shows cost of skipping those heads.")
    (fw2, _), _ = cv2.getTextSize(note, FONT, 0.34, 1)
    cv2.putText(card, note, ((cw - fw2) // 2, card_h - 14),
                FONT, 0.34, (70, 70, 88), 1, cv2.LINE_AA)

    scale_c  = h / card_h
    new_w    = int(w * scale_c)
    resized  = cv2.resize(card, (new_w, h), interpolation=cv2.INTER_AREA)
    out_card = np.zeros((h, w, 3), dtype=np.uint8)
    out_card[:, :new_w] = resized
    return out_card

#  CSV EXPORT

def export_per_second_data(per_second_data, output_path):
    """
    Export per-second proximity data to CSV file.

    Args:
        per_second_data: List of per-second records from compute_proximity_signals
        output_path: Path for the CSV file (e.g., "analysis_per_second.csv")
    """
    import csv

    if not per_second_data:
        print("[WARN] No per-second data to export")
        return

    # CSV headers
    fieldnames = [
        "timestamp",
        "timestamp_unix",
        "signal",
        "distance_m",
        "closure_rate_ms",
        "mmsi_a",
        "mmsi_b"
    ]

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_second_data)

        print(f"[INFO] Per-second data exported to: {output_path}")
        print(f"[INFO] Records exported: {len(per_second_data)}")

    except Exception as e:
        print(f"[ERROR] Failed to export per-second data: {e}")

#  MAIN INFERENCE LOOP

def run(weights, source, csv_path,
        output=None, conf=0.25, iou=0.45,
        img_size=640, summary_frames=90, yaml_cfg="yolov8s.yaml",
        video_start_utc=None,
        dist_threshold=DEFAULT_DIST_THRESHOLD,
        rate_threshold=DEFAULT_RATE_THRESHOLD):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading model  : {weights}")
    model = load_model(weights, yaml_cfg)

    # ── 2. Build early-exit and raw modules ────────────────────────────────────
    layers   = list(model.model.model)
    neck_idx = trace_head_path(layers, 0)

    ee_p3 = EarlyExitModel(layers, neck_idx, 0).to(DEVICE).eval()
    print("[INFO] P3 early-exit module built and ready")

    # FullRawModel: used on P3-signal frames to obtain the baseline P3 raw
    # outputs that are then decoded via decode_p3 — same path as ee_p3.
    # This guarantees P3 counts are identical in routed vs baseline strategies.
    full_raw = FullRawModel(layers).to(DEVICE).eval()
    print("[INFO] FullRawModel built for consistent baseline P3 counting")

    # ── 3. Rigorous pre-loop profiling ─────────────────────────────────────────
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
    frame_signals, prox_summary, per_second_data = compute_proximity_signals(
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
          f"(full {full_ms:.2f}ms vs P3 {p3_ms:.2f}ms, no NMS)")
    print(f"[INFO] Consistency fix: baseline P3 counts via decode_p3 on FullRawModel "
          f"outputs (same decode path as EarlyExitModel)\n")

    # ── 6. Frame loop ──────────────────────────────────────────────────────────
    #
    # Counting strategy:
    #
    #   routed_head_counts   — detections from the signal-chosen path.
    #
    #   baseline_head_counts — detections from always-full model, every frame.
    #
    #   CONSISTENCY GUARANTEE for P3 on P3-signal frames:
    #     Both routed and baseline P3 counts come from decode_p3() applied to
    #     the SAME raw (cv2, cv3) tensors from the P3 head — routed via ee_p3,
    #     baseline via full_raw[0].  The underlying weights are identical, so
    #     the raw tensors are identical and decode_p3 produces identical results.
    #     Therefore Δ P3 = 0 on every P3-signal frame; any non-zero Δ reflects
    #     only missing P4/P5 detections, which is the true cost of routing.
    #
    #   On FULL-signal frames the routed path IS the full model, so we reuse
    #   its results for both counters — no extra run needed.

    routed_head_counts   = {0: 0, 1: 0, 2: 0}
    baseline_head_counts = {0: 0, 1: 0, 2: 0}
    # NEW: P3-specific class breakdown
    p3_class_counts      = {cls: 0 for cls in CLASSES.keys()}
    signal_counts        = {"FULL": 0, "P3": 0}
    frame_no             = 0
    t_wall_start         = time.perf_counter()
    fps_live             = src_fps

    # Initialize P3 tracker
    p3_tracker = SimpleTracker(max_distance=80)

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
            tensor, scale, pad_x, pad_y = preprocess_frame(frame, img_size, DEVICE)
            with torch.no_grad():
                raw_heads = full_raw(tensor)
            head_detections = decode_all_heads(
                raw_heads, img_size, src_w, src_h,
                scale, pad_x, pad_y, conf, iou
            )
            gpu_sync()
            fps_live = 1000.0 / max((time.perf_counter() - t0) * 1000, 1e-3)

            frame, frame_head_counts, frame_class_counts = draw_detections_from_heads(
                frame, head_detections, src_w, src_h)
            active_head = earliest_head_needed(frame_head_counts)

            for h in range(3):
                routed_head_counts[h]   += frame_head_counts.get(h, 0)
                baseline_head_counts[h] += frame_head_counts.get(h, 0)

        else:
            gpu_sync()
            t0 = time.perf_counter()
            tensor, scale, pad_x, pad_y = preprocess_frame(frame, img_size, DEVICE)
            with torch.no_grad():
                cv2_out, cv3_out = ee_p3(tensor)
            routed_p3_dets = decode_p3(
                cv2_out, cv3_out, img_size, src_w, src_h,
                scale, pad_x, pad_y, conf, iou)
            gpu_sync()
            fps_live = 1000.0 / max((time.perf_counter() - t0) * 1000, 1e-3)

            tracked_dets = p3_tracker.update(routed_p3_dets)

            frame, frame_head_counts, frame_class_counts = draw_detections_from_list(
                frame, tracked_dets, src_w, src_h, show_tracking=True)
            active_head = earliest_head_needed(frame_head_counts)

            for h in range(3):
                routed_head_counts[h] += frame_head_counts.get(h, 0)

            # Accumulate P3 class counts
            for cls, count in frame_class_counts.items():
                p3_class_counts[cls] += count

            # ── Baseline: consistent decode_all_heads for all heads ────────────
            # Now that FULL path uses FullRawModel + decode_all_heads, baseline
            # should use IDENTICAL logic for perfect consistency.
            #
            # Both routed (P3-only) and baseline (always-full) now use the same
            # decode functions (decode_p3, decode_p4, decode_p5), eliminating 
            # any attribution inconsistencies between P3 model and full model.

            with torch.no_grad():
                tensor_b, scale_b, pad_x_b, pad_y_b = preprocess_frame(
                    bgr, img_size, DEVICE)
                baseline_raw_heads = full_raw(tensor_b)
                baseline_head_detections = decode_all_heads(
                    baseline_raw_heads, img_size, src_w, src_h,
                    scale_b, pad_x_b, pad_y_b, conf, iou
                )

            # Count detections per head from baseline decode_all_heads
            # All heads (P3, P4, P5) use identical decode logic with their respective strides
            baseline_frame_counts = {
                0: len(baseline_head_detections.get(0, [])),  # P3 via decode_p3
                1: len(baseline_head_detections.get(1, [])),  # P4 via decode_p4
                2: len(baseline_head_detections.get(2, [])),  # P5 via decode_p5
            }
            for h in range(3):
                baseline_head_counts[h] += baseline_frame_counts[h]

        # HUD: fixed profiled speedup + live FPS of active path
        # On P3 frames, show class breakdown and active tracks
        current_p3_classes = frame_class_counts if signal == "P3" else None
        current_tracks = p3_tracker.tracks if signal == "P3" else None

        frame = draw_hud(frame, frame_no, routed_head_counts, fps_live,
                         active_head, signal, speedup,
                         current_p3_classes, current_tracks)
        writer.write(frame)

        if frame_no % 50 == 0 or frame_no == 1:
            pct = frame_no / max(total_f, 1) * 100
            print(f"  [{frame_no:>6}/{total_f}]  {pct:5.1f}%  "
                  f"signal={signal}  fps_live={fps_live:5.1f}  "
                  f"speedup={speedup:.2f}x  "
                  f"routed={dict(routed_head_counts)}  "
                  f"baseline={dict(baseline_head_counts)}")

    duration_s = frame_no / src_fps
    cap.release()

    # ── 7. Summary card ────────────────────────────────────────────────────────
    print(f"\n[INFO] Appending summary card ({summary_frames} frames) ...")
    card = build_summary_card(
        src_h, src_w,
        routed_head_counts, baseline_head_counts,
        frame_no, duration_s,
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

    routed_total   = sum(routed_head_counts.values())
    baseline_total = sum(baseline_head_counts.values())

    print(f"\n{'='*72}")
    print(f"  FINAL REPORT")
    print(f"{'='*72}")
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
    print(f"  -- Head distribution comparison ---------------")
    print(f"  Consistency: All heads (P3/P4/P5) use identical decode functions")
    print(f"  (decode_p3/decode_p4/decode_p5) for both P3 model and full model.")
    print(f"  P3 baseline uses decode_p3 on FullRawModel → Δ P3 = 0 on P3-signal frames.")
    print(f"  P4/P5 baseline uses decode_p4/decode_p5 → consistent attribution.")
    print(f"  Non-zero Δ P4/P5 reflects true cost of skipping those heads.")
    print(f"  {'Head':<16}  {'Baseline':>10}  {'%':>6}  "
          f"{'Routed':>10}  {'%':>6}  {'Δ dets':>8}")
    print(f"  {'-'*64}")
    for h in range(3):
        b_cnt = baseline_head_counts[h]
        r_cnt = routed_head_counts[h]
        b_pct = b_cnt / max(baseline_total, 1) * 100
        r_pct = r_cnt / max(routed_total,   1) * 100
        delta = r_cnt - b_cnt
        print(f"  {HEAD_NAMES[h]:<16}  {b_cnt:>10d}  {b_pct:>5.1f}%  "
              f"{r_cnt:>10d}  {r_pct:>5.1f}%  {delta:>+8d}")
    print(f"  {'-'*64}")
    print(f"  {'Total':<16}  {baseline_total:>10d}  {'100.0%':>6}  "
          f"{routed_total:>10d}  {'100.0%':>6}  "
          f"{routed_total - baseline_total:>+8d}")
    print()
    print(f"  -- P3 class breakdown (P3-signal frames only) -")
    total_p3_dets = sum(p3_class_counts.values())
    if total_p3_dets > 0:
        print(f"  Total P3 detections          : {total_p3_dets}")
        for cls, name in CLASSES.items():
            count = p3_class_counts[cls]
            pct = count / total_p3_dets * 100
            print(f"  {name.capitalize():<16}         : {count:>4d} ({pct:5.1f}%)")
    else:
        print(f"  No P3 detections recorded")
    print(f"  Max simultaneous tracks      : {p3_tracker.next_id - 1}")
    print()
    print(f"  Output saved to               : {output}")
    csv_output = str(Path(output).with_suffix("")) + "_per_second.csv"
    export_per_second_data(per_second_data, csv_output)
    print(f"{'='*72}\n")

    return output, routed_head_counts, baseline_head_counts, full_ms, p3_ms, speedup


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="YOLOv8 inference with proximity-signal routing "
                    "and rigorous pre-loop speedup profiling"
    )
    p.add_argument("--weights",        default="C:/Users/gr3g2/Desktop/trajectories_experiments/models/nano_lr.pt")
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