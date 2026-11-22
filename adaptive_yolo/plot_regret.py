"""
Cumulative-regret figures (seaborn) from `adaptive_yolo.regret` output — SMARTCOMP'26 definition.

    python -m adaptive_yolo.plot_regret results/ground_level_0148-0351 --models 8n 8s 8m

Per model → <dir>/regret/<model>_regret.pdf/.png: cumulative regret vs time step, one line per
target τ (sns.relplot, ordinal blue ramp), one panel per reward setting (adapted / literal).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from .plotting import AXIS, RAMP5, model_name, save, theme

PANEL = {
    "adapted": "Reward = (1 − miss) − 1 · latency / latency(FULL)",
    "literal": "Reward = (1 − miss) − (0.01 / 3) · exit index",
}


def plot_model(df: pd.DataFrame, model: str, out: Path):
    long = pd.concat([
        pd.DataFrame({"step": df["step"], "τ": df["target"].map(lambda t: f"{t:g}"),
                      "cumulative regret": df[f"cum_regret_{v}"], "reward": v,
                      "optimal": df[f"optimal_arm_{v}"]})
        for v in PANEL], ignore_index=True)
    taus = sorted(long["τ"].unique(), key=float)
    g = sns.relplot(data=long, x="step", y="cumulative regret", hue="τ", hue_order=taus,
                    palette=RAMP5[:len(taus)], col="reward", col_order=list(PANEL), kind="line",
                    linewidth=2, height=4.6, aspect=1.25, facet_kws={"sharey": False}, errorbar=None)
    g.set_titles(template="")  # replace seaborn's "reward = …" facet titles with descriptive ones
    for ax, v in zip(g.axes.flat, PANEL):
        best = long.loc[long["reward"] == v, "optimal"].iloc[0].replace("_", " + ")
        ax.set_title(f"{PANEL[v]}\noptimal arm (highest mean reward): {best}", loc="left")
        ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
        ax.set_xlabel("Time step (frame)")
    g.axes.flat[0].set_ylabel("Cumulative regret")
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(taus), title="target τ",
                    frameon=False)
    g.figure.suptitle(f"Cumulative regret of the UCB router  —  {model_name(model)}",
                      x=0.03, ha="left", y=0.985)
    g.figure.text(0.03, 0.905, "instant regret = mean reward of the optimal arm − reward of the arm chosen at "
                               "that step; every frame counted, random FULL probes included")
    g.figure.subplots_adjust(top=0.72, bottom=0.21, left=0.07, right=0.98, wspace=0.18)
    save(g.figure, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--models", nargs="+", default=["8n", "8s", "8m"])
    a = p.parse_args(argv)
    theme()
    res = Path(a.results) / "regret"
    for m in a.models:
        plot_model(pd.read_csv(res / f"{m}_regret.csv"), m, res / f"{m}_regret")
        print(f"{m}: → {res / f'{m}_regret'}.pdf/.png")


if __name__ == "__main__":
    main()
