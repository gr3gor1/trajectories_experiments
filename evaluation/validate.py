import argparse
import time
import csv
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.conv import Conv

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[WARN] thop not found — FLOPs will be skipped. pip install thop")

torch.serialization.add_safe_globals([DetectionModel])

NC        = 2
CLASSES   = {0: "asv", 1: "boat"}
DEVICE    = "cuda"
IMG_SIZE  = 640
CONF      = 0.25
IOU       = 0.45
SPLIT     = "val"
N_WARMUP  = 30
N_TIMING  = 100
SEED      = 42

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(SEED)

HEAD_NAMES = {0: "P3 (small)", 1: "P4 (medium)", 2: "P5 (large)"}

# ── Architecture patching ──────────────────────────────────────────────────────
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
    print(f"[INFO] cv3 heads patched to depthwise-separable for {detect.nl} scales")


def _is_depthwise(state):
    return "model.22.cv3.0.0.0.conv.weight" in state

# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(weights, yaml_cfg="yolov8n.yaml"):
    raw = torch.load(weights, map_location=DEVICE, weights_only=False)

    if isinstance(raw, dict) and "model" in raw and not any(
        k.startswith("model.") for k in raw.keys()
    ):
        inner = raw["model"]
        if isinstance(inner, nn.Module):
            state = inner.state_dict()
        else:
            state = dict(inner)
        print(f"[INFO] Unwrapped training checkpoint "
              f"(epoch={raw.get('epoch', '?')}, "
              f"best_fitness={raw.get('best_fitness', '?')})")
    else:
        state = dict(raw)

    state = {k: v for k, v in state.items()
             if not k.endswith((".total_ops", ".total_params"))}

    det = DetectionModel(yaml_cfg, nc=NC, verbose=False)

    if _is_depthwise(state):
        print("[INFO] Depthwise cv3 detected — patching architecture before loading")
        _patch_cv3_depthwise(det, state)
    else:
        print("[INFO] Standard cv3 heads detected")

    missing, unexpected = det.load_state_dict(state, strict=False)

    real_missing    = [k for k in missing    if not k.endswith((".total_ops", ".total_params"))]
    real_unexpected = [k for k in unexpected if not k.endswith((".total_ops", ".total_params"))]

    if real_missing:
        raise RuntimeError(f"[ERROR] Weights missing from checkpoint:\n{real_missing}")
    if real_unexpected:
        print(f"[WARN] Unexpected keys in checkpoint (ignored):\n{real_unexpected}")

    det.to(DEVICE).eval()

    yolo       = YOLO(yaml_cfg)
    yolo.model = det
    return yolo

# ── Head masking ───────────────────────────────────────────────────────────────
class NegInfClassHead(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, x):
        return torch.full_like(self.orig(x), float("-inf"))


def mask_heads(model, active):
    detect, saved = model.model.model[-1], {}
    for i in range(detect.nl):
        if i != active:
            saved[i] = detect.cv3[i]
            detect.cv3[i] = NegInfClassHead(detect.cv3[i]).to(DEVICE)
            print(f"[INFO] Head {i} ({HEAD_NAMES[i]}) — suppressed")
        else:
            print(f"[INFO] Head {i} ({HEAD_NAMES[i]}) — ACTIVE")
    return saved


def restore_heads(model, saved):
    detect = model.model.model[-1]
    for i, m in saved.items():
        detect.cv3[i] = m

# ── Graph tracing ──────────────────────────────────────────────────────────────
def trace_head_path(layers, head_idx):
    detect_idx = len(layers) - 1
    detect     = layers[detect_idx]

    det_f = detect.f
    if isinstance(det_f, int):
        det_f = [det_f]
    det_f     = [detect_idx - 1 if f == -1 else f for f in det_f]
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

# ── Raw forward-only models (no NMS / decode) ──────────────────────────────────
# FIX: Both baseline and early-exit models are raw backbone+neck+head-branch
# forward passes only. NMS/decode is excluded from both sides so the
# comparison is apples-to-apples.

class FullForwardModel(nn.Module):
    """
    Runs the complete backbone + neck + ALL three head branches (cv2+cv3).
    No postprocessing / NMS. This is the correct baseline to compare against
    the early-exit model, because both stop at the same logical boundary.
    """
    def __init__(self, layers):
        super().__init__()
        self.backbone_neck = nn.ModuleList(layers[:-1])   # everything before Detect
        self.layer_f       = [getattr(l, 'f', -1) for l in layers[:-1]]
        detect             = layers[-1]
        self.cv2           = detect.cv2   # nn.ModuleList of 3
        self.cv3           = detect.cv3   # nn.ModuleList of 3
        self.det_f         = detect.f     # which backbone indices feed the detector

    def forward(self, x):
        n   = len(self.backbone_neck)
        cache = {}

        def resolve(fi, i):
            idx = i - 1 if fi == -1 else fi
            return x if idx < 0 else cache[idx]

        for i, layer in enumerate(self.backbone_neck):
            f   = self.layer_f[i]
            src = [resolve(fi, i) for fi in f] if isinstance(f, list) else resolve(f, i)
            cache[i] = layer(src) if isinstance(src, list) else layer(src)

        # Resolve which cached tensors feed the Detect layer
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
    Runs ONLY the backbone+neck layers required for the active head's feature
    map, then runs that head's cv2+cv3 branches. No postprocessing / NMS.

    Scope matches FullForwardModel: both stop after cv2/cv3 branches,
    before any decoding or NMS — making speedup and FLOPs savings valid.
    """
    def __init__(self, layers, neck_indices, head_idx):
        super().__init__()
        self.exit   = max(neck_indices)
        self.head   = head_idx
        self.needed = set(neck_indices)

        self.layer_map = nn.ModuleDict({str(i): layers[i] for i in neck_indices})
        self.layer_f   = {i: getattr(layers[i], 'f', -1) for i in neck_indices}

        detect   = layers[-1]
        self.cv2 = detect.cv2[head_idx]
        self.cv3 = detect.cv3[head_idx]

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

# ── Timing & FLOPs helpers ─────────────────────────────────────────────────────
def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def measure_flops(module, dummy):
    if not HAS_THOP:
        return None, None

    with torch.no_grad():
        macs, params = thop_profile(module, inputs=(dummy,), verbose=False)

    for submodule in module.modules():
        submodule._forward_hooks.clear()
        for attr in ("total_ops", "total_params"):
            try:
                delattr(submodule, attr)
            except AttributeError:
                pass

    return macs * 2 / 1e9, params


def measure_time(module, dummy, n_warmup=N_WARMUP, n_runs=N_TIMING):
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

    times  = torch.tensor(times)
    mean   = times.mean().item()
    std    = times.std().item()
    median = times.median().item()
    print(f"  [timing]  mean={mean:.2f}ms  median={median:.2f}ms  std={std:.2f}ms  "
          f"cv={std/mean*100:.1f}%")
    return median

# ── Shared baseline ────────────────────────────────────────────────────────────
class SharedBaseline:
    """
    Full-model forward pass (backbone + neck + all 3 head branches) measured
    once, WITHOUT NMS/decode, and reused for all per-head comparisons.
    This ensures the speedup ratio isolates the cost of the unused
    backbone/neck layers only — not NMS or decode overhead.
    """
    def __init__(self, model, dummy):
        layers     = list(model.model.model)
        full_model = FullForwardModel(layers).to(DEVICE)

        print("\n[BASELINE] Measuring full forward pass (no NMS) — shared across all heads")
        self.latency_ms     = measure_time(full_model, dummy)
        self.gflops, self.params = measure_flops(full_model, dummy)

        print(f"\n{'─'*62}")
        print(f"  Full model baseline — backbone+neck+all heads (no NMS)")
        print(f"{'─'*62}")
        print(f"  Full forward latency : {self.latency_ms:.2f} ms")
        if self.gflops:
            print(f"  Full model GFLOPs    : {self.gflops:.3f}")
        if self.params:
            print(f"  Full model params    : {self.params/1e6:.2f}M")
        print(f"{'─'*62}\n")

# ── Detection counter ──────────────────────────────────────────────────────────
@contextmanager
def count_detections_ctx():
    total = [0]
    _orig = DetectionValidator.update_metrics

    def patched(self, preds, batch):
        for pred in preds:
            try:
                if isinstance(pred, torch.Tensor):
                    if pred.ndim == 2 and pred.shape[1] >= 5:
                        total[0] += int((pred[:, 4] > 0.0).sum().item())
                    elif pred.ndim == 1:
                        total[0] += 1 if (len(pred) >= 5 and pred[4] > 0.0) else 0
                elif isinstance(pred, dict):
                    scores = pred.get("conf", pred.get("scores", None))
                    if scores is not None:
                        total[0] += int((scores > 0.0).sum().item())
                    else:
                        boxes = pred.get("boxes", pred.get("bboxes", None))
                        if boxes is not None:
                            total[0] += len(boxes)
            except Exception as e:
                print(f"[WARN] count_detections_ctx: could not parse pred ({type(pred)}): {e}")
        return _orig(self, preds, batch)

    DetectionValidator.update_metrics = patched
    try:
        yield total
    finally:
        DetectionValidator.update_metrics = _orig

# ── Per-head evaluation ────────────────────────────────────────────────────────
def run(weights, data, active_head, baseline,
        split=SPLIT, save_json=False, save_plots=True, output=None):

    print(f"\n[INFO] Loading weights: {weights}")
    model  = load_model(weights)
    output = output or str(Path(weights).parent / f"val_head{active_head}")
    layers = list(model.model.model)

    # 1. Graph analysis
    neck_idx = trace_head_path(layers, active_head)
    print(f"[GRAPH] Head {active_head} ({HEAD_NAMES[active_head]}) requires "
          f"{len(neck_idx)} backbone+neck layers  "
          f"(idx {neck_idx[0]} → {neck_idx[-1]})")

    # 2. Early-exit timing & FLOPs
    # FIX: EarlyExitModel scope = partial backbone+neck + 1 head branch (no NMS)
    #      FullForwardModel scope = full backbone+neck + all head branches (no NMS)
    #      Both exclude NMS/decode → speedup and FLOPs savings are now valid.
    dummy    = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    ee_model = EarlyExitModel(layers, neck_idx, active_head).to(DEVICE)

    full_ms     = baseline.latency_ms
    full_gflops = baseline.gflops
    full_par    = baseline.params

    proj_ms        = measure_time(ee_model, dummy)
    proj_gflops, _ = measure_flops(ee_model, dummy)

    speedup      = full_ms / proj_ms                      if proj_ms                     else None
    flops_saving = (1 - proj_gflops / full_gflops) * 100 if full_gflops and proj_gflops else None

    print(f"\n{'─'*62}")
    print(f"  Early-exit projection — Head {active_head} ({HEAD_NAMES[active_head]})")
    print(f"  (both sides: backbone+neck+head branches only, no NMS/decode)")
    print(f"{'─'*62}")
    print(f"  Full forward latency : {full_ms:.2f} ms  (shared baseline)")
    print(f"  Early-exit latency   : {proj_ms:.2f} ms")
    if speedup is not None:
        print(f"  Speedup              : {speedup:.2f}×")
    if full_gflops and proj_gflops:
        print(f"  Full model GFLOPs    : {full_gflops:.3f}")
        print(f"  Early-exit GFLOPs    : {proj_gflops:.3f}")
        print(f"  FLOPs saved          : {flops_saving:.1f}%")
    if full_par:
        print(f"  Full model params    : {full_par/1e6:.2f}M")
    print(f"{'─'*62}\n")

    # 3. Validation (masked) + detection counting
    saved_cv3 = mask_heads(model, active_head)

    with count_detections_ctx() as total_dets:
        metrics = model.val(
            data      = data,
            imgsz     = IMG_SIZE,
            conf      = CONF,
            iou       = IOU,
            device    = DEVICE,
            split     = split,
            save_json = save_json,
            plots     = save_plots,
            project   = str(Path(output).parent),
            name      = Path(output).name,
            classes   = list(CLASSES.keys()),
            verbose   = False,
        )

    restore_heads(model, saved_cv3)

    # 4. Print results
    print(f"\n{'─'*62}")
    print(f"  Validation — Head {active_head} ({HEAD_NAMES[active_head]})")
    print(f"{'─'*62}")
    print(f"  Total detections     : {total_dets[0]}")
    print(f"  mAP@50               : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95            : {metrics.box.map:.4f}")
    print(f"  Precision            : {metrics.box.mp:.4f}")
    print(f"  Recall               : {metrics.box.mr:.4f}")
    for cls_idx, cls_name in CLASSES.items():
        try:
            print(f"  [{cls_name:>4}]  "
                  f"P={metrics.box.p[cls_idx]:.3f}  "
                  f"R={metrics.box.r[cls_idx]:.3f}  "
                  f"AP@50={metrics.box.ap[cls_idx]:.3f}")
        except IndexError:
            pass
    print(f"{'─'*62}")
    print(f"\n✅  Results saved to: {output}")

    # 5. Build result row for CSV
    row = {
        "head_idx":         active_head,
        "head_name":        HEAD_NAMES[active_head],
        "full_latency_ms":  round(full_ms, 4),
        "ee_latency_ms":    round(proj_ms, 4),
        "speedup":          round(speedup, 4)        if speedup      is not None else "",
        "full_gflops":      round(full_gflops, 4)    if full_gflops  is not None else "",
        "ee_gflops":        round(proj_gflops, 4)    if proj_gflops  is not None else "",
        "flops_saved_pct":  round(flops_saving, 2)   if flops_saving is not None else "",
        "full_params_M":    round(full_par / 1e6, 4) if full_par     is not None else "",
        "total_detections": total_dets[0],
        "map50":            round(metrics.box.map50, 4),
        "map50_95":         round(metrics.box.map,   4),
        "precision":        round(metrics.box.mp,    4),
        "recall":           round(metrics.box.mr,    4),
    }
    for cls_idx, cls_name in CLASSES.items():
        try:
            row[f"{cls_name}_P"]    = round(metrics.box.p[cls_idx],  4)
            row[f"{cls_name}_R"]    = round(metrics.box.r[cls_idx],  4)
            row[f"{cls_name}_AP50"] = round(metrics.box.ap[cls_idx], 4)
        except IndexError:
            row[f"{cls_name}_P"] = row[f"{cls_name}_R"] = row[f"{cls_name}_AP50"] = ""

    return metrics, row

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=False, default="C:/Users/gr3g2/Desktop/trajectories_experiments/models/nano_lr.pt")
    p.add_argument("--data",    required=False, default="C:/Users/gr3g2/Downloads/output_dataset_v3-20260318T064630Z-1-001/output_dataset_v3/data_filtered.yaml")
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--no-plots",  action="store_true")
    p.add_argument("--output",    default=None)
    p.add_argument("--heads", nargs="+", type=int, default=[0, 1, 2], choices=[0, 1, 2],
                   help="Which heads to evaluate (default: all three)")
    args = p.parse_args()

    # Measure full model ONCE before the per-head loop
    # FIX: SharedBaseline now uses FullForwardModel (no NMS) instead of model.model
    print(f"\n[INFO] Loading weights for baseline: {args.weights}")
    baseline_model = load_model(args.weights)
    dummy          = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    baseline       = SharedBaseline(baseline_model, dummy)
    del baseline_model

    all_rows = []
    for head in args.heads:
        output = args.output or str(Path(args.weights).parent / f"val_head{head}")
        _, row = run(
            args.weights, args.data, head,
            baseline,
            args.split, args.save_json, not args.no_plots, output,
        )
        all_rows.append(row)

    csv_path = Path(args.weights).parent / "nano_lr.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n📊  CSV saved to: {csv_path}")