"""
Adaptive YOLOv8 detector with three exit levels.

    ExitLevel.P3     backbone/neck layers needed for P3 + P3 head      (cheapest)
    ExitLevel.P3_P4  layers needed for P3 and P4 + both heads
    ExitLevel.FULL   whole network + P3/P4/P5 heads                     (= plain YOLOv8)

All levels share the same weights (the sub-networks hold references to the
modules of one loaded DetectionModel) and the same decode + NMS code, so
detections from different levels are directly comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as tv_ops
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])


class ExitLevel(IntEnum):
    P3 = 0
    P3_P4 = 1
    FULL = 2


LEVEL_HEADS: dict[ExitLevel, tuple[int, ...]] = {
    ExitLevel.P3: (0,),
    ExitLevel.P3_P4: (0, 1),
    ExitLevel.FULL: (0, 1, 2),
}
HEAD_NAMES = {0: "P3", 1: "P4", 2: "P5"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    head: int  # 0=P3, 1=P4, 2=P5 — the head whose anchor produced the box


# ══════════════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════════════

_SCALE_BY_WIDTH = {16: "n", 32: "s", 48: "m", 64: "l", 80: "x"}


def _model_from_state_dict(state: dict) -> DetectionModel:
    """Rebuild a YOLOv8 DetectionModel from a bare state dict."""
    state = {k: v for k, v in state.items() if not k.endswith((".total_ops", ".total_params"))}
    width = state["model.0.conv.weight"].shape[0]
    scale = _SCALE_BY_WIDTH.get(width)
    if scale is None:
        raise ValueError(f"Cannot infer YOLOv8 scale from stem width {width}")
    nc = state["model.22.cv3.0.2.weight"].shape[0]
    det = DetectionModel(f"yolov8{scale}.yaml", nc=nc, verbose=False)
    det.load_state_dict(state, strict=True)
    return det


def load_detection_model(weights: str | Path, device: str | torch.device = "cpu") -> DetectionModel:
    """Load an Ultralytics checkpoint (.pt with a pickled model) or a bare state dict."""
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        det = ckpt
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("ema"), nn.Module):
        det = ckpt["ema"]
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
        det = ckpt["model"]
    elif isinstance(ckpt, dict):
        det = _model_from_state_dict(ckpt.get("model", ckpt))
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    return det.float().to(device).eval()


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-network for a subset of heads
# ══════════════════════════════════════════════════════════════════════════════

def _parents(layer, i: int) -> list[int]:
    f = getattr(layer, "f", -1)
    f = [f] if isinstance(f, int) else list(f)
    return [i - 1 if fi == -1 else fi for fi in f]


def head_inputs(layers: Sequence[nn.Module]) -> list[int]:
    """Indices of the neck layers that feed the P3/P4/P5 heads."""
    return _parents(layers[-1], len(layers) - 1)


def trace_layers(layers: Sequence[nn.Module], head_idxs: Sequence[int]) -> list[int]:
    """All backbone/neck layer indices needed to compute the given heads' inputs."""
    roots = head_inputs(layers)
    needed: set[int] = set()
    queue = [roots[h] for h in head_idxs]
    while queue:
        i = queue.pop()
        if i < 0 or i in needed:
            continue
        needed.add(i)
        queue.extend(_parents(layers[i], i))
    return sorted(needed)


class PartialNet(nn.Module):
    """Runs only the layers needed for `head_idxs`; returns raw (box, cls) maps per head."""

    def __init__(self, layers: Sequence[nn.Module], head_idxs: Sequence[int]):
        super().__init__()
        self.head_idxs = tuple(head_idxs)
        self.order = trace_layers(layers, head_idxs)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in self.order})
        self.parents = {i: _parents(layers[i], i) for i in self.order}
        self.is_multi = {i: isinstance(getattr(layers[i], "f", -1), list) for i in self.order}
        detect = layers[-1]
        roots = head_inputs(layers)
        self.roots = [roots[h] for h in self.head_idxs]
        self.cv2 = nn.ModuleList(detect.cv2[h] for h in self.head_idxs)
        self.cv3 = nn.ModuleList(detect.cv3[h] for h in self.head_idxs)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cache: dict[int, torch.Tensor] = {}
        for i in self.order:
            srcs = [x if p < 0 else cache[p] for p in self.parents[i]]
            cache[i] = self.layers[str(i)](srcs if self.is_multi[i] else srcs[0])
        return [(b(cache[r]), c(cache[r])) for r, b, c in zip(self.roots, self.cv2, self.cv3)]


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-processing, decode, NMS
# ══════════════════════════════════════════════════════════════════════════════

def letterbox(bgr: np.ndarray, imgsz: int, stride: int = 32, rect: bool = False):
    """Resize keeping aspect ratio and pad (value 114) — same as Ultralytics.

    rect=False pads to imgsz x imgsz (as in the paper's timings);
    rect=True pads only to the next multiple of `stride` (fewer pixels, faster).
    """
    h0, w0 = bgr.shape[:2]
    r = imgsz / max(h0, w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    if rect:
        th, tw = int(np.ceil(nh / stride) * stride), int(np.ceil(nw / stride) * stride)
    else:
        th = tw = imgsz
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def decode_heads(raw, head_idxs, strides) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw head maps → (boxes xyxy [N,4], class scores [N,nc], head id [N]) in input pixels."""
    all_boxes, all_scores, all_heads = [], [], []
    for (box_map, cls_map), h in zip(raw, head_idxs):
        _, c, fh, fw = box_map.shape
        reg_max = c // 4
        stride = float(strides[h])
        dev = box_map.device
        sy, sx = torch.meshgrid(torch.arange(fh, device=dev) + 0.5,
                                torch.arange(fw, device=dev) + 0.5, indexing="ij")
        ax, ay = sx.flatten(), sy.flatten()

        # DFL: channels are laid out [side][bin]; expectation over bins
        dist = box_map[0].permute(1, 2, 0).reshape(-1, 4, reg_max).softmax(-1)
        dist = (dist * torch.arange(reg_max, device=dev, dtype=dist.dtype)).sum(-1)
        boxes = torch.stack([ax - dist[:, 0], ay - dist[:, 1],
                             ax + dist[:, 2], ay + dist[:, 3]], dim=1) * stride

        scores = cls_map[0].permute(1, 2, 0).reshape(fh * fw, -1).sigmoid()
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_heads.append(torch.full((fh * fw,), h, device=dev, dtype=torch.long))
    return torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_heads)


def nms(boxes, scores, heads, conf: float, iou: float, max_det: int = 300):
    """Class-aware NMS over the anchors of all active heads (like Ultralytics)."""
    best, cls = scores.max(dim=1)
    keep = best > conf
    boxes, best, cls, heads = boxes[keep], best[keep], cls[keep], heads[keep]
    if boxes.numel() == 0:
        return boxes, best, cls, heads
    k = tv_ops.batched_nms(boxes, best, cls, iou)[:max_det]
    return boxes[k], best[k], cls[k], heads[k]


# ══════════════════════════════════════════════════════════════════════════════
#  Public model
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveYOLO:
    """
    YOLOv8 detector whose compute is chosen per call.

        model = AdaptiveYOLO("models/nano_lr.pt")
        dets  = model(frame_bgr, ExitLevel.P3)       # cheap
        dets  = model(frame_bgr, ExitLevel.FULL)     # identical to plain YOLOv8

    Pair it with `GradedRouter` (router.py) to pick the level from trajectory data.
    """

    def __init__(self, weights: str | Path, device: str | None = None, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, rect: bool = False, fuse: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_detection_model(weights, self.device)
        if fuse:
            self.model.fuse(verbose=False)
        self.names: dict[int, str] = dict(getattr(self.model, "names", {}) or {})
        self.imgsz, self.conf, self.iou, self.rect = imgsz, conf, iou, rect

        layers = list(self.model.model)
        detect = layers[-1]
        if len(detect.cv2) != 3:
            raise ValueError("Expected a 3-head (P3/P4/P5) YOLOv8 detector")
        self.strides = [float(s) for s in detect.stride]
        self.nets = {lvl: PartialNet(layers, LEVEL_HEADS[lvl]).to(self.device).eval()
                     for lvl in ExitLevel}

    # ── inference ──────────────────────────────────────────────────────────────
    def preprocess(self, bgr: np.ndarray):
        img, r, px, py = letterbox(bgr, self.imgsz, int(max(self.strides)), self.rect)
        t = torch.from_numpy(img[..., ::-1].copy()).to(self.device)  # BGR → RGB
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return t, r, px, py

    @torch.inference_mode()
    def forward_raw(self, x: torch.Tensor, level: ExitLevel):
        return self.nets[ExitLevel(level)](x)

    def _postprocess(self, raw, head_idxs, r, px, py, h0, w0) -> list[Detection]:
        boxes, scores, heads = decode_heads(raw, head_idxs, self.strides)
        boxes, conf, cls, heads = nms(boxes, scores, heads, self.conf, self.iou)
        boxes = boxes.clone()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / r).clamp(0, w0)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / r).clamp(0, h0)
        return [Detection(*b, conf=c, cls=int(k), head=int(h))
                for b, c, k, h in zip(boxes.tolist(), conf.tolist(), cls.tolist(), heads.tolist())]

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray, level: ExitLevel = ExitLevel.FULL) -> list[Detection]:
        level = ExitLevel(level)
        h0, w0 = bgr.shape[:2]
        x, r, px, py = self.preprocess(bgr)
        raw = self.nets[level](x)
        return self._postprocess(raw, LEVEL_HEADS[level], r, px, py, h0, w0)

    @torch.inference_mode()
    def detect_all_levels(self, bgr: np.ndarray) -> dict[ExitLevel, list[Detection]]:
        """
        One FULL forward pass, decoded three ways.

        The P3 / P3_P4 sub-networks compute exactly the same layers as the full
        network up to their heads, so decoding only those heads from the FULL raw
        outputs gives what the cheaper levels *would* have returned on this frame,
        without running them. Costs one FULL forward + three decodes/NMS.
        """
        h0, w0 = bgr.shape[:2]
        x, r, px, py = self.preprocess(bgr)
        raw = self.nets[ExitLevel.FULL](x)  # one (box, cls) pair per head: P3, P4, P5
        return {lvl: self._postprocess(raw[:len(LEVEL_HEADS[lvl])], LEVEL_HEADS[lvl], r, px, py, h0, w0)
                for lvl in ExitLevel}

    # ── profiling ──────────────────────────────────────────────────────────────
    def sync(self) -> None:
        """Wait for queued GPU work, so wall-clock timings cover it."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def warmup(self, shape: tuple[int, int] | None = None, n: int = 10) -> None:
        """Run every level a few times so CUDA init/cudnn setup is not timed."""
        h, w = shape or (self.imgsz, self.imgsz)
        dummy = torch.zeros(1, 3, h, w, device=self.device)
        for net in self.nets.values():
            for _ in range(n):
                net(dummy)
        self.sync()

    @torch.inference_mode()
    def profile(self, warmup: int = 30, runs: int = 100, shape: tuple[int, int] | None = None
                ) -> dict[ExitLevel, float]:
        """
        Median forward latency (ms, no decode/NMS) per level.

        Levels are timed round-robin (P3, P3_P4, FULL, P3, …) rather than one
        after another, so clock/thermal/power-state changes — common on laptop
        GPUs — hit every level equally and the ratios stay meaningful.
        """
        h, w = shape or (self.imgsz, self.imgsz)
        dummy = torch.zeros(1, 3, h, w, device=self.device)
        for _ in range(warmup):
            for net in self.nets.values():
                net(dummy)
        times = {lvl: [] for lvl in self.nets}
        for _ in range(runs):
            for lvl, net in self.nets.items():
                self.sync()
                t0 = time.perf_counter()
                net(dummy)
                self.sync()
                times[lvl].append((time.perf_counter() - t0) * 1000)
        return {lvl: float(np.median(t)) for lvl, t in times.items()}
