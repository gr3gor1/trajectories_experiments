"""
Learned, online routing: which heads to run, learned from the trajectory.

Problem
-------
Online we only see the output of the level we ran, so we cannot tell whether a
cheap level was "enough" on a frame — unless that frame ran FULL. A FULL pass,
however, contains the P3 and P4 head outputs, so `AdaptiveYOLO.detect_all_levels`
gives the exact output of *every* level from one FULL forward. FULL frames are
therefore fully-labelled training examples, and the task becomes a contextual
bandit with free counterfactuals on FULL frames:

    context  x = trajectory features (distance, 1/distance, closure rate, time-to-contact …)
    label    y_L = 1 if level L misses any FULL detection (same class, IoU >= 0.5)
    model    one online Bayesian logistic regression per cheap level: P(miss_L | x)
    decision cheapest level whose miss risk <= target_miss; otherwise FULL

Uncertainty drives exploration: with `strategy="ucb"` the risk is an upper
confidence bound, with `"thompson"` it is a posterior sample, so an untrained
router starts at FULL (risk 0.5) and only steps down where it has evidence.
A small `probe_rate` of random FULL frames keeps checking contexts the policy no
longer visits on its own (otherwise a wrong "P3 is fine here" belief would never
be corrected). Labels depend only on x, never on the decision, so learning only
from FULL frames does not bias P(miss | x) for a well-specified model.

The posterior is a diagonal Gaussian updated per example (Laplace/ADF step, as in
Chapelle & Li, "An Empirical Evaluation of Thompson Sampling", NeurIPS 2011).
`forget` < 1 relaxes the precision towards the prior every update, so the router
keeps adapting if the scene, camera or weather changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import Detection, ExitLevel
from .trajectory import ProximityReading

CHEAP_LEVELS = (ExitLevel.P3, ExitLevel.P3_P4)  # in increasing cost


# ══════════════════════════════════════════════════════════════════════════════
#  Features and labels
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = ("bias", "proximity", "log_dist", "dist_100m", "rate", "approach", "urgency")


def trajectory_features(r: ProximityReading) -> np.ndarray:
    """Fixed-scale features (all roughly in [-1, 1]); no normaliser state to learn."""
    d = max(r.distance_m, 0.0)
    v = r.closure_rate_ms
    approach = max(-v, 0.0)
    ttc = min(d / approach, 120.0) if approach > 0.05 else 120.0  # time to contact, s
    return np.array([
        1.0,
        10.0 / (d + 10.0),          # proximity of the two vessels: 1 at contact, → 0 far apart
        math.log1p(d) / 5.0,
        min(d, 300.0) / 100.0,
        max(min(v, 10.0), -10.0) / 5.0,
        min(approach, 10.0) / 5.0,
        1.0 - ttc / 120.0,           # 1 = contact imminent, 0 = not approaching
    ])


def is_own_vessel(d: Detection, frame_h: int, bottom_frac: float | None) -> bool:
    """Camera mounted on a vessel: its own hull shows up as a box touching the bottom edge."""
    return bottom_frac is not None and d.y2 >= bottom_frac * frame_h


def drop_own_vessel(dets, frame_h: int, bottom_frac: float | None):
    """Remove own-hull boxes from a detection list or from a {level: list} dict."""
    if bottom_frac is None:
        return dets
    if isinstance(dets, dict):
        return {k: drop_own_vessel(v, frame_h, bottom_frac) for k, v in dets.items()}
    return [d for d in dets if not is_own_vessel(d, frame_h, bottom_frac)]


def box_iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def count_missed(reference: list[Detection], candidate: list[Detection],
                 iou_thr: float = 0.5, min_conf: float = 0.0) -> tuple[int, int]:
    """(#reference detections with conf >= min_conf, #of those with no same-class
    candidate at IoU >= iou_thr). Greedy matching, highest confidence first."""
    ref = sorted((d for d in reference if d.conf >= min_conf), key=lambda d: -d.conf)
    used = [False] * len(candidate)
    missed = 0
    for g in ref:
        best, best_j = iou_thr, -1
        for j, c in enumerate(candidate):
            if used[j] or c.cls != g.cls:
                continue
            iou = box_iou(g, c)
            if iou >= best:
                best, best_j = iou, j
        if best_j < 0:
            missed += 1
        else:
            used[best_j] = True
    return len(ref), missed


# ══════════════════════════════════════════════════════════════════════════════
#  Online Bayesian logistic regression (diagonal Gaussian posterior)
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))


class OnlineBayesLogistic:
    def __init__(self, dim: int, prior_var: float = 4.0, forget: float = 1.0):
        self.q0 = 1.0 / prior_var
        self.m = np.zeros(dim)              # posterior mean
        self.q = np.full(dim, self.q0)      # posterior precision (diagonal)
        self.forget = forget
        self.n = 0

    def _moments(self, x):
        return float(self.m @ x), float((x * x / self.q).sum())

    def mean(self, x) -> float:
        mu, s = self._moments(x)            # probit approximation of the predictive
        return _sigmoid(mu / math.sqrt(1.0 + math.pi * s / 8.0))

    def ucb(self, x, beta: float) -> float:
        mu, s = self._moments(x)
        return _sigmoid(mu + beta * math.sqrt(s))

    def sample(self, x, rng: np.random.Generator) -> float:
        w = self.m + rng.standard_normal(self.m.shape) / np.sqrt(self.q)
        return _sigmoid(float(w @ x))

    def update(self, x, y: float) -> None:
        if self.forget < 1.0:               # drift back towards the prior → keeps adapting
            self.q = self.q0 + self.forget * (self.q - self.q0)
        p = _sigmoid(float(self.m @ x))
        self.q = self.q + p * (1.0 - p) * x * x
        self.m = self.m + (y - p) * x / self.q
        self.n += 1


# ══════════════════════════════════════════════════════════════════════════════
#  Router
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedRouter:
    """
    Usage per frame:

        level = router(reading)
        if level == ExitLevel.FULL:
            all_dets = model.detect_all_levels(frame)   # free counterfactuals
            router.observe(all_dets)
            dets = all_dets[ExitLevel.FULL]
        else:
            dets = model(frame, level)
    """
    target_miss: float = 0.10       # acceptable P(cheap level misses something FULL finds)
    strategy: str = "ucb"           # "ucb" | "thompson" | "greedy"
    beta: float = 1.0               # UCB width (std devs of the logit)
    probe_rate: float = 0.05        # share of random FULL frames (exploration)
    forget: float = 0.999
    prior_var: float = 4.0
    label_conf: float = 0.25        # FULL detections below this are ignored as labels
    label_iou: float = 0.5
    fallback: ExitLevel = ExitLevel.FULL
    seed: int = 0

    models: dict = field(init=False, repr=False)
    last_risk: dict = field(init=False, repr=False)
    last_probe: bool = field(default=False, init=False)
    _x: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.strategy not in ("ucb", "thompson", "greedy"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        self.models = {lvl: OnlineBayesLogistic(len(FEATURE_NAMES), self.prior_var, self.forget)
                       for lvl in CHEAP_LEVELS}
        self.rng = np.random.default_rng(self.seed)
        self.last_risk, self.last_probe, self._x = {}, False, None

    def risk(self, level: ExitLevel, x: np.ndarray) -> float:
        m = self.models[level]
        if self.strategy == "ucb":
            return m.ucb(x, self.beta)
        if self.strategy == "thompson":
            return m.sample(x, self.rng)
        return m.mean(x)

    def __call__(self, reading: ProximityReading | None) -> ExitLevel:
        self.last_probe, self.last_risk = False, {}
        if reading is None:
            self._x = None
            return self.fallback
        self._x = trajectory_features(reading)
        self.last_risk = {lvl: self.risk(lvl, self._x) for lvl in CHEAP_LEVELS}
        if self.rng.random() < self.probe_rate:
            self.last_probe = True
            return ExitLevel.FULL
        for lvl in CHEAP_LEVELS:
            if self.last_risk[lvl] <= self.target_miss:
                return lvl
        return ExitLevel.FULL

    def observe(self, all_dets: dict[ExitLevel, list[Detection]]) -> dict[ExitLevel, int]:
        """Learn from a FULL frame (needs every level's detections). Returns the labels."""
        if self._x is None:
            return {}
        labels = {}
        for lvl in CHEAP_LEVELS:
            _, missed = count_missed(all_dets[ExitLevel.FULL], all_dets[lvl],
                                     self.label_iou, self.label_conf)
            labels[lvl] = int(missed > 0)
            self.models[lvl].update(self._x, labels[lvl])
        return labels

    def describe(self) -> str:
        """Learned weights per cheap level (positive weight → more misses)."""
        rows = [f"{'':10s}" + "".join(f"{n:>11s}" for n in FEATURE_NAMES) + "   n"]
        for lvl, m in self.models.items():
            rows.append(f"{lvl.name:10s}" + "".join(f"{w:11.2f}" for w in m.m) + f"   {m.n}")
        return "\n".join(rows)
