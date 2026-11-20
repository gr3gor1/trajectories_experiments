"""
Compare routing criteria on actual video frames.

    python -m adaptive_yolo.compare_frames results/ground_level_0148-0351 --model 8n \
        --source GROUND_LEVEL.MP4 --own-vessel-bottom 0.97 --both-asvs

Picks moments where the criteria disagree (different levels or different misses),
spread over the window, and renders a seaborn FacetGrid: one row per moment, one
column per criterion. Every tile in a row shows the same crop of the real frame with

    solid blue box     a full-model detection the criterion's level also found
    dashed red box     a full-model detection that level missed
    header             the level the criterion ran and how many it missed

With --both-asvs only frames where the full model detects both logged ASVs (yellow and
dark green, see find_both_asvs) are eligible, and the crop spans the two.

Outputs <dir>/frames/<model>_criteria_frames.pdf/.png, plus one full-resolution
PNG per moment and criterion under <dir>/frames/<model>/.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .learned import box_iou, drop_own_vessel, is_own_vessel
from .model import ExitLevel
from .plotting import (CRITICAL, FOUND, LEVEL_COLOUR, LEVEL_LABEL, model_name, policy_title, save,
                       theme)

DEFAULT_CRITERIA = ["heuristic_binary_30m", "heuristic_graded_30_60m", "learned_ucb_t0.3", "oracle"]


def split_found(full, cand, iou_thr=0.5, min_conf=0.25):
    """Full-model detections (conf >= min_conf) split into found / missed by `cand`."""
    ref = sorted((d for d in full if d.conf >= min_conf), key=lambda d: -d.conf)
    used, found, missed = [False] * len(cand), [], []
    for g in ref:
        best, bj = iou_thr, -1
        for j, c in enumerate(cand):
            if not used[j] and c.cls == g.cls and box_iou(g, c) >= best:
                best, bj = box_iou(g, c), j
        (missed if bj < 0 else found).append(g)
        if bj >= 0:
            used[bj] = True
    return found, missed


def pick_moments(frames: pd.DataFrame, criteria: list[str], n: int, eligible: np.ndarray | None = None,
                 bonus: np.ndarray | None = None) -> list[int]:
    """Row indices: disagreement-rich frames, one per time segment (only `eligible` frames)."""
    lv = frames[criteria]
    miss = np.stack([np.where(lv[c] == "P3", frames.miss_P3 > 0,
                              np.where(lv[c] == "P3_P4", frames.miss_P3_P4 > 0, False)) for c in criteria], 1)
    n_levels = lv.nunique(axis=1).to_numpy()
    n_miss = miss.sum(1)
    score = (n_levels - 1) + 1.5 * ((n_miss > 0) & (n_miss < len(criteria))) + 0.2 * np.minimum(frames.n_full_dets, 4)
    if bonus is not None:
        score = score + bonus
    ok = (frames.n_full_dets > 0).to_numpy().copy()
    if eligible is not None:
        ok = ok & eligible
    score = np.where(ok, score, -np.inf)
    picks = []
    for seg in np.array_split(np.arange(len(frames)), n):
        if np.isfinite(score[seg]).any():
            picks.append(int(seg[np.argmax(score[seg])]))
    return picks


def vessel_colour(hsv: np.ndarray, d) -> str | None:
    """'yellow' / 'green' from the median HSV of the box centre (OpenCV units), else None.

    The two logged vessels in GROUND_LEVEL are a yellow and a dark-green ASV; colour tells
    them apart from each other and from other boats the detector also calls 'asv'."""
    w, h = d.x2 - d.x1, d.y2 - d.y1
    if w < 60 or h > w:  # too small to identify, or not vessel-shaped
        return None
    x1, y1 = int(d.x1 + 0.25 * w), int(d.y1 + 0.25 * h)
    x2, y2 = max(x1 + 1, int(d.x2 - 0.25 * w)), max(y1 + 1, int(d.y2 - 0.25 * h))
    p = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(int)
    hh, ss, vv = p[:, 0], p[:, 1], p[:, 2]
    # Share of hull-coloured pixels, not a median: hue is circular, so the median of a red hull
    # (pixels near 0 and near 180) can land anywhere, e.g. in the yellow band.
    yellow = ((hh >= 20) & (hh <= 35) & (ss >= 120) & (vv >= 120)).mean()
    green = ((hh >= 75) & (hh <= 110) & (ss >= 60) & (vv <= 90)).mean()
    if yellow >= 0.30 and yellow > green:
        return "yellow"
    if green >= 0.35 and green > yellow:
        return "green"
    return None


def find_both_asvs(cap, cache, H, own_bottom, label_conf, size_band=(0.5, 2.0), asv_class=0):
    """
    Per cached frame: {'yellow': det, 'green': det} for the logged ASVs found by the full model.

    0. class: the detector must call it 'asv';
    1. colour: share of yellow / dark-green hull pixels in the box centre (vessel_colour);
    2. shape: height 25–90 % of width, not cut off by the left/right frame edge;
    3. size consistency: the camera ASV follows both vessels at their speed, so each keeps
       roughly the same apparent size — boxes outside size_band × the colour's median width
       over the whole window are rejected (removes small/large look-alikes).
    """
    f0 = cache.get("start_frame", 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    per_frame = []                       # [(colour, det), ...] per frame
    for _, dets in cache["frames"]:
        ok, img = cap.read()
        if not ok:
            break
        cands = [d for d in dets[ExitLevel.FULL]
                 if d.cls == asv_class and d.conf >= label_conf and not is_own_vessel(d, H, own_bottom)
                 and 0.25 <= (d.y2 - d.y1) / max(d.x2 - d.x1, 1) <= 0.9
                 and d.x1 > 2 and d.x2 < W - 2]
        tagged = []
        if len(cands) >= 2:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            tagged = [(c, d) for d in cands if (c := vessel_colour(hsv, d))]
        per_frame.append(tagged)

    widths = {c: [d.x2 - d.x1 for f in per_frame for cc, d in f if cc == c] for c in ("yellow", "green")}
    median = {c: float(np.median(w)) for c, w in widths.items() if w}
    print("  typical box width (px): " + ", ".join(f"{c} {m:.0f} (n={len(widths[c])})" for c, m in median.items()))
    found = []
    for tagged in per_frame:
        ident = {}
        for c, d in sorted(tagged, key=lambda t: -t[1].conf):
            m = median.get(c)
            if c not in ident and m and size_band[0] * m <= d.x2 - d.x1 <= size_band[1] * m:
                ident[c] = d
        found.append(ident if len(ident) == 2 else {})
    return found


def union_crop(boxes, w, h, aspect, pad=0.08):
    """Smallest crop of the given aspect containing all boxes (+ padding), clipped to the frame."""
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    bw, bh = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    cw = min(w, max(bw, bh * aspect))
    ch = min(h, cw / aspect)
    cw = ch * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox = int(np.clip(cx - cw / 2, 0, w - cw))
    oy = int(np.clip(cy - ch / 2, 0, h - ch))
    return ox, oy, int(ox + cw), int(oy + ch)


def crop_box(dets, w, h, aspect=2.4, pad=0.25, min_w=0.35):
    """Crop around all detections (same for every tile in a row)."""
    if not dets:
        cw = w * 0.6
        return int(w * 0.2), int(h * 0.35), int(w * 0.2 + cw), int(h * 0.35 + cw / aspect)
    x1, y1 = min(d.x1 for d in dets), min(d.y1 for d in dets)
    x2, y2 = max(d.x2 for d in dets), max(d.y2 for d in dets)
    cw = max((x2 - x1) * (1 + 2 * pad), w * min_w, (y2 - y1) * (1 + 2 * pad) * aspect)
    ch = cw / aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox = int(np.clip(cx - cw / 2, 0, max(0, w - cw)))
    oy = int(np.clip(cy - ch / 2, 0, max(0, h - ch)))
    return ox, oy, int(min(w, ox + cw)), int(min(h, oy + ch))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--model", default="8n")
    p.add_argument("--source", required=True)
    p.add_argument("--criteria", nargs="+", default=DEFAULT_CRITERIA)
    p.add_argument("--moments", type=int, default=6)
    p.add_argument("--own-vessel-bottom", type=float, default=None)
    p.add_argument("--label-conf", type=float, default=0.25)
    p.add_argument("--both-asvs", action="store_true",
                   help="only frames where the full model detects both logged ASVs (identified by hull colour: "
                        "yellow and dark green); the crop spans the two (needs --own-vessel-bottom)")
    a = p.parse_args(argv)
    if a.both_asvs and a.own_vessel_bottom is None:
        p.error("--both-asvs needs --own-vessel-bottom")

    theme()
    res = Path(a.results)
    frames = pd.read_csv(res / f"{a.model}_frames.csv")
    cache = pickle.loads((res / f"{a.model}_cache.pkl").read_bytes())
    fps, f0, H = cache["fps"], cache.get("start_frame", 0), cache.get("frame_h")
    crit = [c for c in a.criteria if c in frames.columns]

    cap = cv2.VideoCapture(a.source)
    eligible = bonus = asvs = None
    if a.both_asvs:
        asvs = find_both_asvs(cap, cache, H, a.own_vessel_bottom, a.label_conf)
        eligible = np.array([len(x) == 2 for x in asvs] + [False] * (len(frames) - len(asvs)))
        # prefer both ASVs clearly visible (the smaller of the two boxes not tiny)
        bonus = np.array([min(min(d.x2 - d.x1 for d in x.values()) / 250, 1.0) if len(x) == 2 else 0
                          for x in asvs] + [0] * (len(frames) - len(asvs)))
        print(f"{a.model}: {eligible.sum()} of {len(eligible)} frames show both logged ASVs (green + yellow)")
    picks = pick_moments(frames, crit, a.moments, eligible, bonus)
    out_dir = res / "frames" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    col_order = ["FULL model (reference)"] + [policy_title(c)[0] for c in crit]
    tiles = []  # one record per (moment, criterion) → tidy DataFrame for sns.FacetGrid

    for i in picks:
        reading, dets = cache["frames"][i]
        frame_no = f0 + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"cannot read frame {frame_no}")
        h, w = img.shape[:2]
        dets = drop_own_vessel(dets, H or h, a.own_vessel_bottom)
        full = [d for d in dets[ExitLevel.FULL] if d.conf >= a.label_conf]
        if a.both_asvs:  # crop around the two logged ASVs
            pair = list(asvs[i].values())
            x1, y1, x2, y2 = union_crop([(d.x1, d.y1, d.x2, d.y2) for d in pair], w, h, aspect=2.0, pad=0.35)
        else:
            x1, y1, x2, y2 = crop_box(full, w, h)
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        t = i / fps
        when = f"t = {t:5.1f} s · {reading.distance_m:.1f} m · {reading.closure_rate_ms:+.2f} m/s" if reading else f"t = {t:.1f} s"

        levels = ["FULL"] + [frames.loc[i, c] for c in crit]
        for c, (name, level) in enumerate(zip(col_order, levels)):
            found, missed = split_found(dets[ExitLevel.FULL], dets[ExitLevel[level]], 0.5, a.label_conf)
            tiles.append({"moment": when, "criterion": name, "level": level, "crop": crop,
                          "extent": (x1, x2, y2, y1), "found": found, "missed": missed, "n_full": len(full),
                          "names": dict(asvs[i]) if a.both_asvs else {}})

            # full-resolution copy of this tile
            tile = img.copy()
            for d in found:
                cv2.rectangle(tile, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (214, 120, 42), 5)
            for d in missed:
                cv2.rectangle(tile, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (59, 59, 208), 5)
            tag = "full" if c == 0 else crit[c - 1]
            cv2.imwrite(str(out_dir / f"frame{frame_no}_{tag}.png"), tile[y1:y2, x1:x2])
    cap.release()

    df = pd.DataFrame(tiles)
    row_order = list(dict.fromkeys(df["moment"]))
    x1, x2, y2, y1 = df["extent"].iloc[0]
    g = sns.FacetGrid(df, row="moment", col="criterion", row_order=row_order, col_order=col_order,
                      height=1.95, aspect=(x2 - x1) / (y2 - y1), sharex=False, sharey=False, despine=False)
    g.map_dataframe(draw_tile)
    g.set_titles(template="")
    for r, moment in enumerate(row_order):
        for c, name in enumerate(col_order):
            ax = g.axes[r, c]
            rec = df[(df["moment"] == moment) & (df["criterion"] == name)].iloc[0]
            status = (f"missed {len(rec['missed'])} of {rec['n_full']}" if rec["missed"]
                      else f"all {rec['n_full']} found")
            ax.set_title(f"{LEVEL_LABEL[rec['level']]} · {status}", fontsize="x-small", pad=3)
            if r == 0:
                ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(0, 22), textcoords="offset points",
                            fontsize="small", va="bottom")
            if c == 0:
                ax.set_ylabel(moment, fontsize="x-small", rotation=0, ha="right", va="center",
                              labelpad=6)

    handles = [Line2D([], [], color=FOUND, linewidth=2, label="full-model detection, also found"),
               Line2D([], [], color=CRITICAL, linewidth=2, linestyle=(0, (3, 2)), label="full-model detection, missed")]
    handles += [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=LEVEL_COLOUR[l], linewidth=3,
                          label=f"tile frame: {LEVEL_LABEL[l]} ran") for l in ("P3", "P3_P4", "FULL")]
    fig = g.figure
    fig_w, fig_h = fig.get_size_inches()
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False,
               bbox_to_anchor=(0.01, 0.0), columnspacing=2.0)
    fig.suptitle(f"Criteria compared on video frames  —  {model_name(a.model)}", x=0.01, ha="left", y=0.995)
    fig.set_size_inches(fig_w + 1.6, fig_h + 1.4)
    fig_h += 1.4
    fig.subplots_adjust(left=1.75 / (fig_w + 1.6), right=0.995, top=1 - 0.95 / fig_h, bottom=1.0 / fig_h,
                        wspace=0.04, hspace=0.32)
    stem = res / "frames" / f"{a.model}_criteria_frames"
    save(fig, stem, dpi=170)
    print(f"{a.model}: frames {[f0 + i for i in picks]} → {stem}.pdf/.png")


def draw_tile(data: pd.DataFrame, **_):
    """FacetGrid mapper: draw one frame crop with its found / missed boxes into the current facet."""
    rec = data.iloc[0]
    ax = plt.gca()
    x1, x2, y2, y1 = rec["extent"]
    ax.imshow(rec["crop"], extent=rec["extent"])
    for d in rec["found"]:
        ax.add_patch(Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1, fill=False, edgecolor=FOUND, linewidth=1.6))
    for d in rec["missed"]:
        ax.add_patch(Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1, fill=False, edgecolor=CRITICAL,
                               linewidth=1.8, linestyle=(0, (3, 2))))
    for colour, d in rec["names"].items():  # name the two logged vessels
        ax.annotate(f"{colour} ASV", (d.x1, d.y1), xytext=(0, 2), textcoords="offset points", fontsize="xx-small",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.set_aspect("auto")  # the crop already has the facet's aspect; fill the facet, no side gaps
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(LEVEL_COLOUR[rec["level"]])
        s.set_linewidth(3)


if __name__ == "__main__":
    main()
