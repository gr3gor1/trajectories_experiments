# adaptive_yolo

Trajectory-guided adaptive YOLOv8: the distance and closure rate between two
vessels decide how much of the detector runs on each frame.

| Level   | Runs                         | Rule (defaults)                                  |
|---------|------------------------------|--------------------------------------------------|
| `P3`    | layers 0–15 + P3 head        | distance ≥ 60 m                                  |
| `P3_P4` | layers 0–18 + P3, P4 heads   | distance < 60 m                                  |
| `FULL`  | whole network (plain YOLOv8) | distance < 30 m **and** closing faster than 0.5 m/s |

Missing or stale trajectory data falls back to `FULL`. `--mid-dist 0` removes
the middle level, giving the paper's binary P3/FULL rule.

All levels share one set of weights and one decode + NMS implementation. At
`FULL` the output matches Ultralytics' own decode + NMS: the same number of
detections on every image tested, boxes within 0.04 px.

## Modules

| Module | Role |
|---|---|
| `model.py` | `AdaptiveYOLO`: the three exit levels (`ExitLevel`), shared decode/NMS, `detect_all_levels`, profiling |
| `trajectory.py` | `TrajectoryLog` (recorded log → causal per-second readings, video alignment) and `ProximityTracker` (live fixes) |
| `router.py` | `GradedRouter`: the binary (paper) and graded threshold rules, with optional hysteresis |
| `learned.py` | `LearnedRouter`: online Bayesian UCB / Thompson router; miss labels (`count_missed`, `box_iou`); camera-hull filter |
| `run_video.py` | annotate a video with any router; per-frame CSV |
| `evaluate_routers.py` | run the network once per frame, cache it, replay every criterion (`policies`) against the full model |
| `benchmark.py` | measured (wall-clock) speedups of every criterion on the real frames |
| `regret.py` | cumulative regret of the UCB routers (SMARTCOMP'26 definition) |
| `aggregate.py` | results and speedup tables across YOLOv8 scales (Markdown + LaTeX) |
| `plotting.py`, `plot_signals.py`, `plot_regret.py`, `compare_frames.py` | seaborn figures |

Data flows one way: `model` and `trajectory` are the building blocks, the routers only
map a `ProximityReading` to an `ExitLevel`, and every script under them reads the
cached outputs of `evaluate_routers`, so figures and tables can be rebuilt without a GPU.

## Recorded video + log

```bash
python -m adaptive_yolo.run_video --weights models/nano_lr.pt \
    --source clip.mp4 --csv ASV_logs_filtered.csv
```

**Trajectory readings.** The log is split into 1-second slots counted from its
**first record**, and video second *k* (frames `k·fps` … `(k+1)·fps − 1`) uses slot
*k*. The video is assumed to start one second after the log, so slot *k* is complete
when video second *k* starts and nothing uses future data. If a vessel has no fix in
a slot, its last known position is held for up to 2 s; after that there is no reading
and the frame runs FULL. The closure rate is d[k] − d[k−1], and 0 in the first second.

**Alignment.** The default (frame 0 = first record) is correct when the clip and the
log were cut to the same window (e.g. `video_preprocessing.py` 01:48–03:51 with
`preprocessing.py` 11:54:06–11:56:09). Otherwise pair any video position with a log
time, e.g. the uncut recording with `--sync 01:48=11:54:06` (and `--start/--end` to
process only that window), or give `--video-start` as a full UTC timestamp.
`--offset` shifts by extra seconds for fine-tuning. The run prints which log interval
the video maps to and stops if less than half of it has trajectory data
(`--min-coverage`).

The CSV needs `t`, `mmsi` and either float `lon`/`lat` or a WKB `geom` column
(the output of `trajectories/simultaneous_trajectories.py` works as-is).
Outputs: `<clip>_adaptive.mp4` and `<clip>_adaptive_frames.csv` (level,
distance, closure rate, latency and detections per head for every frame), plus
a profiled speedup estimate. Use `--force-level FULL` for the always-full
baseline, `--hold N` to keep a level for at least N frames before stepping
down, and `--rect` to pad to a multiple of 32 instead of 640×640.

## Learned routing (`learned.py`)

Instead of hand-set thresholds, `LearnedRouter` learns online, from the
trajectory, how likely each cheap level is to miss something.

- **Free labels.** One FULL forward pass contains the P3 and P4 head outputs, so
  `model.detect_all_levels(frame)` returns the exact output of all three levels.
  Every FULL frame is therefore a fully labelled example: did P3 / P3_P4 miss any
  FULL detection (same class, IoU ≥ 0.5)?
- **Model.** One online Bayesian logistic regression per cheap level,
  P(miss | trajectory features). The features describe the two tracked vessels:
  proximity 10/(d+10), log distance, distance, closure rate, approach speed and
  time-to-contact urgency. A forgetting factor keeps it adapting.
- **Decision.** Pick the cheapest level whose miss risk is ≤ `target_miss`,
  otherwise FULL. Risk is a UCB (default) or a Thompson sample, so an untrained
  router starts at FULL. A small share of random FULL probe frames (`--probe`)
  keeps it checking; with probe 0 it gets stuck and does worse.

```bash
python -m adaptive_yolo.run_video ... --router learned --target-miss 0.3
```

`target_miss` is the knob for the speed/accuracy trade-off. `router.describe()`
prints the learned weights; a large positive `proximity` weight means "the
vessels are close → the cheap level misses".

## Comparing policies (`evaluate_routers.py`)

```bash
python -m adaptive_yolo.evaluate_routers --weights models/nano_lr.pt --source clip.mp4 --csv log.csv
python -m adaptive_yolo.evaluate_routers --cache clip_routers_cache.pkl --targets 0.2 0.3 0.4 0.5
```

The first command runs the network once per frame (one FULL pass decoded as
every level) and caches the result. Any policy can then be replayed in seconds:
fixed levels, the binary paper rule, the graded heuristic, learned routers at
several targets, and a hindsight oracle as the upper bound. Learned policies
only see labels on frames where they chose FULL, as they would live. Outputs
are the speedup, recall vs FULL, the frame miss rate and the level shares
(`_summary.csv`), plus every policy's choice per frame (`_frames.csv`).

## Live use

```python
from adaptive_yolo import AdaptiveYOLO, GradedRouter, ProximityTracker

model   = AdaptiveYOLO("models/nano_lr.pt")
router  = GradedRouter(full_dist=30, mid_dist=60, closing_rate=0.5, hold_frames=15)
tracker = ProximityTracker(window_s=2.0, max_age_s=5.0)

# whenever a position fix arrives:
tracker.update(mmsi, t_seconds, lon, lat)

# per camera frame:
level = router(tracker.reading(now=t_seconds))
for d in model(frame_bgr, level):
    print(d.cls, d.conf, d.head, d.x1, d.y1, d.x2, d.y2)
```

`model.profile()` returns the median forward latency per level; the levels are timed
round-robin so GPU clock changes affect all of them equally.

## Extending

* **A new routing criterion** is any callable `router(reading) -> ExitLevel`
  (`reading` is a `ProximityReading` or `None`). Add it to `policies()` in
  `evaluate_routers.py` and it is evaluated, benchmarked, tabulated and plotted with
  the others from the existing caches. A router that learns can expose
  `observe(all_levels_detections)`; it is called on frames where it chose FULL.
* **New trajectory features** for the learned router go in `trajectory_features` and
  `FEATURE_NAMES` (`learned.py`).
* **Tests.** `python -m unittest discover -s tests -v` checks the trajectory readings,
  the heuristic rules, matching and the learned router without a GPU or the video.
  Changes should also leave `evaluate_routers --cache …` reproducing the saved
  `results/` CSVs exactly.
