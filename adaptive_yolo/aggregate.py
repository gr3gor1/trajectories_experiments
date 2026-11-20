"""
Aggregate evaluate_routers results across YOLOv8 scales into one table.

    python -m adaptive_yolo.aggregate results/ground_level_0148-0351 --models 8n 8s 8m

Writes, in the results directory:
    aggregate_long.csv   one row per (model, policy) with every metric and the per-level latency
    aggregate_wide.csv   one row per policy, columns <model>_<metric>
    aggregate.md         Markdown tables: speedup / recall / frames with a miss per scale, and — if
                         adaptive_yolo.benchmark has run — projected vs measured speedup
    aggregate.tex        the same as LaTeX tables (booktabs) for the paper
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

ORDER = ["always_FULL", "always_P3_P4", "always_P3", "heuristic_binary_30m", "heuristic_graded_30_60m"]
NAMES = {
    "always_FULL": "Always FULL (baseline)",
    "always_P3_P4": "Always P3+P4",
    "always_P3": "Always P3",
    "heuristic_binary_30m": "Heuristic, binary (paper rule)",
    "heuristic_graded_30_60m": "Heuristic, graded 30/60 m",
    "oracle": "Oracle (hindsight)",
}
METRICS = [("speedup", "Speedup", "{:.2f}×"), ("recall_vs_full", "Recall", "{:.3f}"),
           ("frame_miss", "Miss", "{:.1%}")]


def nice(policy: str) -> str:
    if policy in NAMES:
        return NAMES[policy]
    kind, _, t = policy.rpartition("_t")
    return {"learned_ucb": "Learned UCB", "learned_thompson": "Learned Thompson"}.get(kind, kind) + f", τ={t}"


def model_name(m: str) -> str:
    return f"YOLOv8{m[1:]}" if m.startswith("8") else m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--models", nargs="+", default=["8n", "8s", "8m"])
    a = p.parse_args(argv)
    res = Path(a.results)

    rows, lat = [], {}
    for m in a.models:
        s = pd.read_csv(res / f"{m}_summary.csv")
        s.insert(0, "model", model_name(m))
        cache = pickle.loads((res / f"{m}_cache.pkl").read_bytes())
        lat[m] = cache["latency_ms"]
        for lvl, ms in cache["latency_ms"].items():
            s[f"latency_{lvl}_ms"] = ms
        rows.append(s)
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(res / "aggregate_long.csv", index=False)

    policies = list(dict.fromkeys(
        ORDER + [p for p in long["policy"] if p.startswith("learned_ucb")]
        + [p for p in long["policy"] if p.startswith("learned_thompson")] + ["oracle"]))
    policies = [p for p in policies if p in set(long["policy"])]
    wide = (long.set_index(["policy", "model"])[[k for k, _, _ in METRICS] + ["P3", "P3_P4", "FULL"]]
            .unstack("model"))
    wide.columns = [f"{mdl}_{metric}" for metric, mdl in wide.columns]
    wide = wide.loc[policies]
    wide.to_csv(res / "aggregate_wide.csv")

    models = [model_name(m) for m in a.models]

    def cell(policy, mdl, key, fmt):
        return fmt.format(wide.loc[policy, f"{mdl}_{key}"])

    # Markdown
    head = "| Criterion | " + " | ".join(f"{mdl} {lab}" for mdl in models for _, lab, _ in METRICS) + " |"
    sep = "|---|" + "---:|" * (len(models) * len(METRICS))
    lines = [head, sep]
    for pol in policies:
        lines.append(f"| {nice(pol)} | " + " | ".join(cell(pol, mdl, k, f) for mdl in models for k, _, f in METRICS) + " |")
    lat_line = "Forward latency per level (ms, median): " + "; ".join(
        f"{model_name(m)} P3 {lat[m]['P3']:.2f} / P3+P4 {lat[m]['P3_P4']:.2f} / FULL {lat[m]['FULL']:.2f}" for m in a.models)
    md = "\n".join(lines) + "\n\n" + lat_line + "\n\nSpeedup: projected from profiled forward latency. " \
         "Recall: share of full-model detections (conf ≥ 0.25, own hull excluded) also found. " \
         "Miss: share of frames with ≥ 1 full-model detection where the chosen level missed one.\n"

    # measured speedups (adaptive_yolo.benchmark), if available
    bench = {m: res / "benchmark" / f"{m}_benchmark.csv" for m in a.models}
    bench = {m: pd.read_csv(p).set_index("policy") for m, p in bench.items() if p.exists()}
    speed_tex = []
    if bench:
        bm = [m for m in a.models if m in bench]
        bpol = [p for p in policies if all(p in bench[m].index for m in bm)]
        cols = [("projected_speedup", "Projected"), ("measured_model_speedup", "Measured (model)"),
                ("measured_e2e_speedup", "Measured (end-to-end)")]
        shead = "| Criterion | " + " | ".join(f"{model_name(m)} {lab}" for m in bm for _, lab in cols) + " |"
        ssep = "|---|" + "---:|" * (len(bm) * len(cols))
        slines = [shead, ssep] + [
            f"| {nice(p)} | " + " | ".join(f"{bench[m].loc[p, k]:.2f}×" for m in bm for k, _ in cols) + " |"
            for p in bpol]
        ms_line = "Measured per-frame cost of always-FULL (model / end-to-end, ms): " + "; ".join(
            f"{model_name(m)} {bench[m].loc['always_FULL', 'measured_model_ms_per_frame']:.1f} / "
            f"{bench[m].loc['always_FULL', 'measured_e2e_ms_per_frame']:.1f}" for m in bm if "always_FULL" in bench[m].index)
        md += ("\n## Speedup: projected vs measured\n\n" + "\n".join(slines) + "\n\n" + ms_line +
               "\n\nProjected: profiled forward-pass latency × frames per level (network only). "
               "Measured (model): wall-clock per frame for the level each criterion chose, timed on the same "
               "decoded frame — pre-processing, forward pass, box decoding and NMS, plus the router's own "
               "overhead; learned routers pay the full decode of all levels on their FULL frames. "
               "Measured (end-to-end): additionally includes decoding the 4K video frame. All relative to "
               "always-FULL measured the same way.\n")
        speed_tex = [r"\begin{table*}[t]", r"\centering", r"\small",
                     r"\begin{tabular}{l" + "rrr" * len(bm) + "}", r"\toprule",
                     " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{model_name(m)}}}" for m in bm) + r" \\",
                     "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(bm))),
                     "Criterion & " + " & ".join(["Proj.", "Meas. (model)", "Meas. (e2e)"] * len(bm)) + r" \\",
                     r"\midrule"]
        for p in bpol:
            name = nice(p).replace("τ", r"$\tau$").replace("×", r"$\times$")
            speed_tex.append(f"{name} & " + " & ".join(f"{bench[m].loc[p, k]:.2f}$\\times$"
                                                        for m in bm for k, _ in cols) + r" \\")
            if p in ("heuristic_graded_30_60m",) or p.startswith("learned_thompson"):
                speed_tex.append(r"\midrule")
        speed_tex += [r"\bottomrule", r"\end{tabular}",
                      r"\caption{Projected vs measured speedup relative to always running the full model. "
                      r"Projected: profiled forward-pass latency only. Measured (model): per-frame wall-clock "
                      r"time of pre-processing, forward pass, decoding and NMS for the level each criterion "
                      r"chose, plus routing overhead. Measured (e2e): additionally includes decoding the "
                      r"4K video frame.}",
                      r"\label{tab:speedup-measured}", r"\end{table*}"]
    (res / "aggregate.md").write_text(md, encoding="utf-8")

    # LaTeX (booktabs)
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\begin{tabular}{l" + "rrr" * len(models) + "}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{mdl}}}" for mdl in models) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models))),
           "Criterion & " + " & ".join(["Speedup", "Recall", "Miss"] * len(models)) + r" \\", r"\midrule"]
    for pol in policies:
        vals = []
        for mdl in models:
            vals += [f"{wide.loc[pol, f'{mdl}_speedup']:.2f}$\\times$",
                     f"{wide.loc[pol, f'{mdl}_recall_vs_full']:.3f}",
                     f"{100 * wide.loc[pol, f'{mdl}_frame_miss']:.1f}\\%"]
        name = nice(pol).replace("τ", r"$\tau$").replace("×", r"$\times$")
        tex.append(f"{name} & " + " & ".join(vals) + r" \\")
        if pol in ("heuristic_graded_30_60m",) or pol.startswith("learned_thompson"):
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Routing criteria across YOLOv8 scales. Speedup is projected from profiled forward latency "
            r"relative to always running the full model; recall is the share of full-model detections retained; "
            r"miss is the share of frames on which the chosen level missed a full-model detection.}",
            r"\label{tab:routing-criteria}", r"\end{table*}"]
    if speed_tex:
        tex += [""] + speed_tex
    (res / "aggregate.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
