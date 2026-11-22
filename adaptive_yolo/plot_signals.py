"""
Signal-classification diagrams (seaborn), one per routing criterion, from evaluate_routers output.

    python -m adaptive_yolo.plot_signals results/ground_level_0148-0351 --models 8n 8s 8m

Per model and criterion → <dir>/signals/<model>/<criterion>.pdf/.png:

    top     distance between the vessels per second (sns.scatterplot, hue = level the
            criterion ran in that second, majority over its frames)
    middle  closure rate per second, same encoding
    bottom  sns.heatmap strips: the level on every frame, and frames where it missed
            a full-model detection

Per model → <dir>/signals/<model>_overview.pdf/.png: distance trace (sns.lineplot) over two
heatmaps with one row per criterion — level per frame, and frames with a miss.

Distance and closure rate are separate panels on a shared time axis (no dual y-axis).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plotting import (AXIS, CRITICAL, INK, INK_2, LEVEL_COLOUR, LEVEL_LABEL, LEVELS, MUTED, SURFACE,
                       model_name, policy_title, save, theme)

SKIP = {"always_P3", "always_P3_P4", "always_FULL"}
LEVEL_CMAP = ListedColormap([LEVEL_COLOUR[l] for l in LEVELS])
MISS_CMAP = ListedColormap([SURFACE, CRITICAL])
STRIP_DPI = 450  # the strips need at least one pixel per frame, or single-frame events are dropped


def load(results: Path, model: str):
    frames = pd.read_csv(results / f"{model}_frames.csv")
    summary = pd.read_csv(results / f"{model}_summary.csv").set_index("policy")
    fps = 29.97
    cache = results / f"{model}_cache.pkl"
    if cache.exists():
        fps = pickle.loads(cache.read_bytes()).get("fps", fps)
    frames["t"] = (frames["frame"] - frames["frame"].iloc[0]) / fps
    frames["sec"] = frames["t"].astype(int)
    return frames, summary, fps


def missed(frames: pd.DataFrame, policy: str) -> np.ndarray:
    lv = frames[policy]
    return np.where(lv == "P3", frames["miss_P3"] > 0, np.where(lv == "P3_P4", frames["miss_P3_P4"] > 0, False))


def per_second(frames: pd.DataFrame, policy: str) -> pd.DataFrame:
    g = frames.groupby("sec")
    out = g.agg(distance=("distance_m", "first"), rate=("closure_rate_ms", "first"))
    out["level"] = g[policy].agg(lambda s: s.value_counts().idxmax())
    out = out.reset_index()
    out["t"] = out["sec"] + 0.5
    return out


def strip(ax, values: np.ndarray, cmap, vmax, ylabels):
    """Heatmap strip(s) over frames. values: [rows, frames]."""
    sns.heatmap(values, ax=ax, cmap=cmap, vmin=0, vmax=vmax, cbar=False, xticklabels=False,
                yticklabels=ylabels, rasterized=True)
    ax.set_xlim(0, values.shape[1])
    ax.tick_params(axis="y", length=0, rotation=0)
    ax.set_xlabel("")


def seconds_axis(ax, n_frames: int, fps: float, step_s: int = 20):
    ticks = np.arange(0, n_frames / fps + 1e-9, step_s)
    ax.set_xticks(ticks * fps)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(axis="x", length=3, color=MUTED, labelbottom=True)


def legend_handles(with_miss=True):
    h = [Patch(facecolor=LEVEL_COLOUR[l], edgecolor="none", label=LEVEL_LABEL[l]) for l in LEVELS]
    if with_miss:
        h.append(Line2D([], [], color=CRITICAL, linewidth=2,
                        label="frame where the chosen level missed a full-model detection"))
    return h


def short_name(policy: str) -> str:
    """Compact row label for the overview heatmaps."""
    if policy.startswith("learned_"):
        kind, _, t = policy.rpartition("_t")
        return f"Learned {'UCB' if kind.endswith('ucb') else 'Thompson'}, τ = {float(t):g}"
    return {"heuristic_binary_30m": "Heuristic, binary (paper)", "heuristic_graded_30_60m": "Heuristic, graded",
            "oracle": "Oracle (hindsight)"}.get(policy, policy_title(policy)[0])


def metrics_line(summary, policy) -> str:
    r = summary.loc[policy]
    return (f"speedup {r['speedup']:.2f}×   ·   P3 {r['P3']:.0%} · P3 + P4 {r['P3_P4']:.0%} · FULL {r['FULL']:.0%} "
            f"of frames   ·   recall vs full model {r['recall_vs_full']:.3f} · frames with a miss {r['frame_miss']:.1%}")


def plot_policy(frames, summary, fps, model, policy, out: Path, xlabel: str):
    sec = per_second(frames, policy)
    title, rule = policy_title(policy)
    n = len(frames)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 2, 0.45, 0.45], hspace=0.12)
    ax_d, ax_r = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_l, ax_m = fig.add_subplot(gs[2]), fig.add_subplot(gs[3])

    for ax, col, ylabel in ((ax_d, "distance", "Distance (m)"), (ax_r, "rate", "Closure rate (m/s)")):
        sns.lineplot(data=sec, x="t", y=col, ax=ax, color=AXIS, linewidth=1.5, zorder=1)
        sns.scatterplot(data=sec, x="t", y=col, hue="level", hue_order=LEVELS, palette=LEVEL_COLOUR, ax=ax,
                        s=46, edgecolor=SURFACE, linewidth=1.5, zorder=3, legend=False)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_xlim(0, n / fps)
        ax.tick_params(labelbottom=False)
        sns.despine(ax=ax)
    ax_r.axhline(0, color=AXIS, linewidth=1, zorder=0)
    ax_d.set_ylim(0, max(45, np.nanmax(sec["distance"]) * 1.1))
    if policy.startswith("heuristic"):  # thresholds this criterion actually uses
        for ax, y, lab in ((ax_d, 30, "30 m"), (ax_r, -0.5, "−0.5 m/s")):
            ax.axhline(y, color=INK_2, linewidth=1, zorder=2)
            ax.annotate(lab, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center")

    lvl = frames[policy].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()[None, :]
    strip(ax_l, lvl, LEVEL_CMAP, 2, ["level"])
    strip(ax_m, missed(frames, policy).astype(int)[None, :], MISS_CMAP, 1, ["miss"])
    ax_l.tick_params(labelbottom=False)
    seconds_axis(ax_m, n, fps)
    ax_m.set_xlabel(xlabel)

    fig.suptitle(f"{title}  —  {model_name(model)}", x=0.07, ha="left", y=0.995)
    fig.text(0.07, 0.938, rule, ha="left")
    fig.text(0.07, 0.908, metrics_line(summary, policy), ha="left")
    fig.legend(handles=legend_handles(), loc="lower left", bbox_to_anchor=(0.065, 0.0), ncol=4,
               handlelength=1.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.885, bottom=0.13)
    save(fig, out, dpi=STRIP_DPI)


def plot_overview(frames, summary, fps, model, policies, out: Path, xlabel: str):
    n, k = len(frames), len(policies)
    names = [f"{short_name(p)}  (recall {summary.loc[p, 'recall_vs_full']:.3f})" for p in policies]
    lv = np.stack([frames[p].map({l: i for i, l in enumerate(LEVELS)}).to_numpy() for p in policies])
    ms = np.stack([missed(frames, p).astype(int) for p in policies])

    fig = plt.figure(figsize=(13, 3.2 + 0.62 * k * 2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 0.48 * k, 0.48 * k], hspace=0.22)
    ax_d, ax_l, ax_m = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    sec = frames.groupby("sec")["distance_m"].first().reset_index()
    sec["frame_x"] = (sec["sec"] + 0.5) * fps
    sns.lineplot(data=sec, x="frame_x", y="distance_m", ax=ax_d, color=INK, linewidth=2)
    ax_d.axhline(30, color=MUTED, linewidth=1)
    ax_d.annotate("30 m", (1.0, 30), xycoords=("axes fraction", "data"), xytext=(4, 0),
                  textcoords="offset points", va="center")
    ax_d.set_xlim(0, n)
    ax_d.set_ylim(0, max(45, sec["distance_m"].max() * 1.1))
    ax_d.set_ylabel("Distance (m)")
    ax_d.set_xlabel("")
    ax_d.tick_params(labelbottom=False)
    sns.despine(ax=ax_d)

    strip(ax_l, lv, LEVEL_CMAP, 2, names)
    ax_l.set_title("Level run on each frame", loc="left")
    ax_l.tick_params(labelbottom=False)
    strip(ax_m, ms, MISS_CMAP, 1, names)
    ax_m.set_title("Frames where the chosen level missed a full-model detection", loc="left")
    for ax in (ax_l, ax_m):  # thin surface gap between criterion rows
        for y in range(1, k):
            ax.axhline(y, color=SURFACE, linewidth=2)
    seconds_axis(ax_m, n, fps)
    ax_m.set_xlabel(xlabel)

    fig.suptitle(f"Level chosen per frame by each criterion  —  {model_name(model)}", x=0.02, ha="left",
                 y=0.995)
    fig.legend(handles=legend_handles(), loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4,
               handlelength=1.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.26, right=0.95, top=0.95, bottom=0.08)
    save(fig, out, dpi=STRIP_DPI)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="directory with <model>_frames.csv / _summary.csv")
    p.add_argument("--models", nargs="+", default=None, help="default: every *_frames.csv in the directory")
    p.add_argument("--xlabel", default="Time in window (s)")
    a = p.parse_args(argv)

    theme()
    res = Path(a.results)
    models = a.models or sorted(f.name[:-len("_frames.csv")] for f in res.glob("*_frames.csv"))
    for model in models:
        frames, summary, fps = load(res, model)
        policies = [c for c in summary.index if c not in SKIP and c in frames.columns]
        for pol in policies:
            plot_policy(frames, summary, fps, model, pol, res / "signals" / model / pol, a.xlabel)
        plot_overview(frames, summary, fps, model, policies, res / "signals" / f"{model}_overview", a.xlabel)
        print(f"{model}: {len(policies)} criteria → {res / 'signals' / model}")


if __name__ == "__main__":
    main()
