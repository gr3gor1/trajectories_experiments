# GROUND_LEVEL experiment: window 01:48–03:51

## Setup

| | |
|---|---|
| Video | `GROUND_LEVEL.MP4`, 3840×2160, 29.97 fps; frames 3237–6922 (3,686 frames, 123 s) |
| Log | `ASV_logs (1).csv`, MMSI 99999 vs 100000, 2025-09-17 |
| Alignment | video 01:48 ↔ log 11:54:06 UTC (`--sync 01:48=11:54:06`), the same pairing as `video_preprocessing.py` / `preprocessing.py` |
| Trajectory | 1 s slots from the first record, causal: last position held ≤ 2 s, closure rate = backward difference |
| Models | `8n.pt`, `8s.pt`, `8m.pt` (YOLOv8 n/s/m, classes asv/boat), imgsz 640, conf 0.25, NMS IoU 0.45 |
| Own vessel | The camera is on a vessel whose bow is always in view. Boxes with bottom edge ≥ 97% of frame height are that hull and are excluded from labels and metrics (`--own-vessel-bottom 0.97`) |
| Reference | Full model (P3+P4+P5) on every frame. A detection (conf ≥ 0.25) counts as found if the chosen level has the same class at IoU ≥ 0.5 |
| Latency | Median forward pass (no decode/NMS), levels timed round-robin, 300 rounds, RTX 4050 Laptop on AC power |
| Learned routers | UCB or Thompson, 5% random FULL probes, seed 0, cold start at the beginning of the window |

## Files

| File | Content |
|---|---|
| `aggregate.md` / `aggregate.tex` | Results table across YOLOv8 scales, plus a projected-vs-measured speedup table (Markdown / LaTeX booktabs) |
| `benchmark/<model>_benchmark.csv` | Measured speedup per criterion: model-only (pre-processing + forward + decode + NMS of the chosen level, plus routing overhead) and end-to-end (+ 4K video decode), vs always-FULL |
| `benchmark/<model>_benchmark_frames.csv` | Per-frame wall-clock of video decode and of each level on that same frame (levels in shuffled order, GPU-synchronised) |
| `aggregate_long.csv`, `aggregate_wide.csv` | The same numbers, machine-readable, plus level shares and per-level latency |
| `<model>_summary.csv` | One row per criterion: speedup, recall, frame miss rate, level shares, probes |
| `<model>_frames.csv` | Per frame: distance, closure rate, number of full-model detections, misses of P3 and P3+P4, and the level every criterion chose |
| `<model>_cache.pkl` | Detections of all three levels for every frame + latency; replay without the GPU via `evaluate_routers --cache` |
| `signals/<model>/<criterion>.pdf/.png` | Signal-classification diagram per criterion (seaborn): distance and closure rate coloured by level, per-frame level and miss strips |
| `signals/<model>_overview.pdf/.png` | All criteria as heatmap rows under the distance trace: level per frame, frames with a miss |
| `frames/<model>_criteria_frames.pdf/.png` | Criteria compared on frames where the full model detects both logged ASVs (green and yellow); found = blue, missed = dashed red. An ASV is identified by class `asv`, ≥ 30–35% hull-coloured pixels, a vessel-like box not cut by the frame edge, and a width within 0.5–2× that ASV's median width (the camera ASV follows both, so their size stays roughly constant) |
| `frames/<model>/frame<N>_<criterion>.png` | Full-resolution crops behind that figure |
| `regret/<model>_regret.pdf/.png` | Cumulative regret of the UCB router as defined in arXiv:2604.24810 (SMARTCOMP'26): instant regret = mean reward of the optimal arm − reward of the chosen arm, summed over frames. Two rewards: adapted (λ = 1, latency cost) and literal (λ = 0.01/3, exit-index cost) |
| `regret/<model>_regret.csv`, `regret/regret_summary.csv` | Per-frame reward regret and decisions; per-run totals, first/last-third regret rates, growth exponent |

All figures are produced with seaborn by the scripts in `adaptive_yolo/` (`plot_signals.py`, `plot_regret.py`,
`compare_frames.py`; shared theme in `plotting.py`).
| `videos/8n_<criterion>.mp4` + `_frames.csv` | Annotated 1080p video of the window, with its per-frame log |

## Reproduce

```bash
# pass 1 (GPU) + replay, per model
python -m adaptive_yolo.evaluate_routers --weights 8n.pt --source GROUND_LEVEL.MP4 --csv "ASV_logs (1).csv" \
    --sync 01:48=11:54:06 --start 01:48 --end 03:51 --own-vessel-bottom 0.97 \
    --targets 0.1 0.2 0.3 0.4 0.5 --out results/ground_level_0148-0351/8n
# re-measure latency into an existing cache (plugged in!)
python -m adaptive_yolo.evaluate_routers --cache results/ground_level_0148-0351/8n_cache.pkl --weights 8n.pt \
    --reprofile --profile-runs 300 --own-vessel-bottom 0.97 --targets 0.1 0.2 0.3 0.4 0.5 --out results/ground_level_0148-0351/8n
# measured speedups (plugged in; ~6 min per model), then the tables
python -m adaptive_yolo.benchmark results/ground_level_0148-0351 --model 8n --weights 8n.pt \
    --source GROUND_LEVEL.MP4 --own-vessel-bottom 0.97
# table, diagrams, frame comparison
python -m adaptive_yolo.aggregate results/ground_level_0148-0351
python -m adaptive_yolo.plot_signals results/ground_level_0148-0351 --xlabel "Seconds since video 01:48 (log 11:54:06 UTC)"
python -m adaptive_yolo.compare_frames results/ground_level_0148-0351 --model 8n --source GROUND_LEVEL.MP4 --own-vessel-bottom 0.97 --both-asvs
# SMARTCOMP'26 regret (data, then figures)
python -m adaptive_yolo.regret results/ground_level_0148-0351 --own-vessel-bottom 0.97
python -m adaptive_yolo.plot_regret results/ground_level_0148-0351
# annotated video (learned router, matches the learned_ucb_t0.3 row)
python -m adaptive_yolo.run_video --weights 8n.pt --source GROUND_LEVEL.MP4 --csv "ASV_logs (1).csv" \
    --sync 01:48=11:54:06 --start 01:48 --end 03:51 --own-vessel-bottom 0.97 --out-width 1920 \
    --router learned --target-miss 0.3 --probe 0.05 --output results/ground_level_0148-0351/videos/8n_learned_ucb_t0.3.mp4
```
