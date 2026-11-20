"""
Cumulative regret of the learned (UCB) router's choices, as defined in
Papanikolaou, Kontopoulos & Tserpes, "A Comparative Analysis on the Performance of Upper
Confidence Bound Algorithms in Adaptive Deep Neural Networks", IEEE SMARTCOMP 2026
(arXiv:2604.24810):

    instant regret_t = (mean reward of the optimal arm) − (reward of the arm chosen at step t)
    cumulative regret R_T = Σ_t regret_t            (one step = one frame; every frame counted)

The paper's reward (Eq. 2) is quality − λ·cost. For a detector, quality = 1 − miss_t(a), where
miss_t(a) = 1 if level a missed a full-model detection on frame t. Two cost settings:

    adapted  cost = latency(a) / latency(FULL), λ = 1          (the router's own trade-off)
    literal  cost = exit index 1/2/3,          λ = 0.01 / 3    (paper: ε = 0.01 over the number of exits)

The mean reward of each arm is taken over the whole run (every arm's outcome on every frame is
known from the cache), and the optimal arm is the one with the highest mean.

    python -m adaptive_yolo.regret results/ground_level_0148-0351 --own-vessel-bottom 0.97

Writes <dir>/regret/<model>_regret.csv (per frame and target) and regret/regret_summary.csv.
Plot with `python -m adaptive_yolo.plot_regret`.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .learned import CHEAP_LEVELS, LearnedRouter, count_missed, drop_own_vessel
from .model import ExitLevel
from .plotting import model_name

REWARDS = {  # name: (λ, cost kind)
    "adapted": (1.0, "latency"),
    "literal": (0.01 / 3, "exit_index"),
}


def prepare(cache: dict, own_vessel_bottom, label_conf):
    frames = cache["frames"]
    if own_vessel_bottom is not None:
        frames = [(r, drop_own_vessel(d, cache["frame_h"], own_vessel_bottom)) for r, d in frames]
    miss = np.zeros((len(frames), 3), dtype=int)  # columns: P3, P3_P4, FULL (FULL never misses)
    for t, (_, dets) in enumerate(frames):
        for j, lvl in enumerate(CHEAP_LEVELS):
            miss[t, j] = count_missed(dets[ExitLevel.FULL], dets[lvl], 0.5, label_conf)[1] > 0
    return frames, miss


def reward_matrix(cache: dict, miss: np.ndarray, name: str) -> np.ndarray:
    lam, kind = REWARDS[name]
    lat = cache["latency_ms"]
    cost = (np.array([lat["P3"], lat["P3_P4"], lat["FULL"]]) / lat["FULL"] if kind == "latency"
            else np.array([1.0, 2.0, 3.0]))
    return (1.0 - miss) - lam * cost[None, :]


def run(cache, frames, miss, target, probe, label_conf, seed) -> pd.DataFrame:
    """Replay the online UCB router (identical to evaluate_routers) and compute paper regret."""
    router = LearnedRouter(target_miss=target, strategy="ucb", probe_rate=probe, label_conf=label_conf, seed=seed)
    chosen, probes = np.zeros(len(frames), int), np.zeros(len(frames), bool)
    for t, (reading, dets) in enumerate(frames):
        lvl = router(reading)
        chosen[t], probes[t] = int(lvl), router.last_probe
        if lvl == ExitLevel.FULL:
            router.observe(dets)

    idx = np.arange(len(frames))
    out = {"frame": cache.get("start_frame", 0) + idx, "step": idx + 1, "target": target,
           "chosen": [ExitLevel(c).name for c in chosen], "probe": probes}
    for name in REWARDS:
        r = reward_matrix(cache, miss, name)
        mu = r.mean(0)
        out[f"regret_{name}"] = mu.max() - r[idx, chosen]
        out[f"cum_regret_{name}"] = np.cumsum(out[f"regret_{name}"])
        out[f"optimal_arm_{name}"] = ExitLevel(int(mu.argmax())).name
    return pd.DataFrame(out)


def growth_exponent(R: np.ndarray) -> float:
    """β in R_t ≈ c·t^β fitted on the second half (β < 1 → sub-linear; NaN if R ≤ 0 there)."""
    t = np.arange(1, len(R) + 1)
    h = slice(len(R) // 2, None)
    if (R[h] <= 0).any():
        return float("nan")
    return float(np.polyfit(np.log(t[h]), np.log(R[h]), 1)[0])


def summarise(df: pd.DataFrame, model: str) -> dict:
    thirds = np.array_split(np.arange(len(df)), 3)
    out = {"model": model_name(model), "target": df["target"].iloc[0], "steps": len(df)}
    for name in REWARDS:
        r = df[f"regret_{name}"].to_numpy()
        out[f"{name}_optimal_arm"] = df[f"optimal_arm_{name}"].iloc[0]
        out[f"{name}_cum_regret"] = r.sum()
        out[f"{name}_regret_per_step_first_third"] = r[thirds[0]].mean()
        out[f"{name}_regret_per_step_last_third"] = r[thirds[2]].mean()
        out[f"{name}_growth_exponent"] = growth_exponent(np.cumsum(r))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--models", nargs="+", default=["8n", "8s", "8m"])
    p.add_argument("--targets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--probe", type=float, default=0.05)
    p.add_argument("--label-conf", type=float, default=0.25)
    p.add_argument("--own-vessel-bottom", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    res = Path(a.results)
    out_dir = res / "regret"
    out_dir.mkdir(exist_ok=True)
    summary = []
    for m in a.models:
        cache = pickle.loads((res / f"{m}_cache.pkl").read_bytes())
        frames, miss = prepare(cache, a.own_vessel_bottom, a.label_conf)
        runs = [run(cache, frames, miss, t, a.probe, a.label_conf, a.seed) for t in a.targets]

        ev_path = res / f"{m}_frames.csv"  # sanity: the replay must reproduce evaluate_routers' decisions
        if ev_path.exists():
            ev = pd.read_csv(ev_path)
            for df in runs:
                col = f"learned_ucb_t{df['target'].iloc[0]:g}"
                if col in ev and (ev[col].to_numpy() != df["chosen"].to_numpy()).any():
                    print(f"[warn] {m} {col}: replay differs from evaluate_routers")

        pd.concat(runs).to_csv(out_dir / f"{m}_regret.csv", index=False)
        summary += [summarise(df, m) for df in runs]

    s = pd.DataFrame(summary)
    s.to_csv(out_dir / "regret_summary.csv", index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
