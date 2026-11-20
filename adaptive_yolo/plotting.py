"""
Shared seaborn theme, colour tokens and labels for every figure under results/.

Typography and layout are seaborn's defaults (sns.set_theme()). Data colours:
* levels are ordinal (P3 < P3+P4 < FULL by cost) → one-hue blue ramp, light → dark
  (steps 250/450/650; monotone lightness, ΔL ≈ 0.19 per step, light end 2.06:1 on the surface);
* targets τ are ordinal too → 5-step blue ramp (250…650, ΔL ≈ 0.095 per step);
* a miss is a state → status "critical" red, always with a text label;
* text uses ink tokens, never series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
FOUND = "#2a78d6"

LEVELS = ["P3", "P3_P4", "FULL"]
LEVEL_LABEL = {"P3": "P3 only", "P3_P4": "P3 + P4", "FULL": "FULL (P3 + P4 + P5)"}
LEVEL_COLOUR = {"P3": "#86b6ef", "P3_P4": "#2a78d6", "FULL": "#104281"}
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

POLICY_TITLE = {
    "heuristic_binary_30m": "Heuristic, binary (paper rule)",
    "heuristic_graded_30_60m": "Heuristic, graded",
    "oracle": "Oracle (hindsight)",
    "always_P3": "Always P3",
    "always_P3_P4": "Always P3 + P4",
    "always_FULL": "Always FULL",
}
POLICY_RULE = {
    "heuristic_binary_30m": "FULL if distance < 30 m and closing faster than 0.5 m/s, otherwise P3",
    "heuristic_graded_30_60m": "FULL if distance < 30 m and closing faster than 0.5 m/s; P3 + P4 if < 60 m; "
                               "otherwise P3",
    "oracle": "cheapest level that misses nothing the full model found (needs hindsight, not deployable)",
}


def theme() -> None:
    """Plain seaborn defaults (style "darkgrid", context "notebook", seaborn's font and sizes).

    Only `pdf.fonttype` is set (embed TrueType so PDF text stays editable); everything typographic
    is seaborn's own. Smaller text in the scripts uses seaborn-relative sizes ("small", "x-small").
    """
    sns.set_theme(rc={"pdf.fonttype": 42})


def model_name(m: str) -> str:
    """'8n' → 'YOLOv8n'; anything else unchanged."""
    return f"YOLOv{m}" if m[:1].isdigit() else m


def policy_title(name: str) -> tuple[str, str]:
    """(title, one-line rule) for a policy id from evaluate_routers."""
    if name in POLICY_TITLE:
        return POLICY_TITLE[name], POLICY_RULE.get(name, "")
    kind, _, t = name.rpartition("_t")
    strat = {"learned_ucb": "UCB", "learned_thompson": "Thompson sampling"}.get(kind, kind)
    return (f"Learned router ({strat}), target {float(t):g}",
            f"cheapest level whose estimated miss risk ≤ {float(t):g}; learns online from FULL frames")


def save(fig, stem: Path, dpi: int = 160) -> None:
    """Write <stem>.pdf and <stem>.png (stem may contain dots, e.g. learned_ucb_t0.3)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.parent / (stem.name + ".pdf"))
    fig.savefig(stem.parent / (stem.name + ".png"), dpi=dpi)
    plt.close(fig)
