| Criterion | YOLOv8n Speedup | YOLOv8n Recall | YOLOv8n Miss | YOLOv8s Speedup | YOLOv8s Recall | YOLOv8s Miss | YOLOv8m Speedup | YOLOv8m Recall | YOLOv8m Miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always FULL (baseline) | 1.00× | 1.000 | 0.0% | 1.00× | 1.000 | 0.0% | 1.00× | 1.000 | 0.0% |
| Always P3+P4 | 1.21× | 0.998 | 0.4% | 1.16× | 0.997 | 0.8% | 1.16× | 0.999 | 0.1% |
| Always P3 | 1.50× | 0.854 | 33.8% | 1.38× | 0.864 | 32.9% | 1.32× | 0.922 | 17.4% |
| Heuristic, binary (paper rule) | 1.36× | 0.904 | 22.0% | 1.28× | 0.902 | 23.4% | 1.24× | 0.945 | 12.2% |
| Heuristic, graded 30/60 m | 1.16× | 0.999 | 0.1% | 1.12× | 0.997 | 0.7% | 1.13× | 0.999 | 0.1% |
| Learned UCB, τ=0.1 | 1.19× | 0.998 | 0.4% | 1.15× | 0.997 | 0.7% | 1.16× | 0.999 | 0.3% |
| Learned UCB, τ=0.2 | 1.23× | 0.983 | 4.2% | 1.15× | 0.995 | 1.3% | 1.20× | 0.985 | 3.1% |
| Learned UCB, τ=0.3 | 1.31× | 0.954 | 10.8% | 1.20× | 0.972 | 6.6% | 1.24× | 0.969 | 6.9% |
| Learned UCB, τ=0.4 | 1.37× | 0.921 | 17.9% | 1.27× | 0.935 | 15.4% | 1.27× | 0.957 | 9.6% |
| Learned UCB, τ=0.5 | 1.38× | 0.918 | 18.9% | 1.30× | 0.916 | 19.7% | 1.28× | 0.951 | 11.0% |
| Learned Thompson, τ=0.3 | 1.35× | 0.933 | 15.8% | 1.27× | 0.933 | 15.8% | 1.27× | 0.950 | 11.1% |
| Oracle (hindsight) | 1.39× | 1.000 | 0.0% | 1.30× | 1.000 | 0.0% | 1.29× | 1.000 | 0.0% |

Forward latency per level (ms, median): YOLOv8n P3 3.37 / P3+P4 4.17 / FULL 5.04; YOLOv8s P3 3.93 / P3+P4 4.68 / FULL 5.42; YOLOv8m P3 11.16 / P3+P4 12.65 / FULL 14.71

Speedup: projected from profiled forward latency. Recall: share of full-model detections (conf ≥ 0.25, own hull excluded) also found. Miss: share of frames with ≥ 1 full-model detection where the chosen level missed one.

## Speedup: projected vs measured

| Criterion | YOLOv8n Projected | YOLOv8n Measured (model) | YOLOv8n Measured (end-to-end) | YOLOv8s Projected | YOLOv8s Measured (model) | YOLOv8s Measured (end-to-end) | YOLOv8m Projected | YOLOv8m Measured (model) | YOLOv8m Measured (end-to-end) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always FULL (baseline) | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| Always P3+P4 | 1.21× | 1.15× | 1.07× | 1.16× | 1.15× | 1.07× | 1.16× | 1.13× | 1.07× |
| Always P3 | 1.50× | 1.36× | 1.16× | 1.38× | 1.33× | 1.13× | 1.32× | 1.26× | 1.13× |
| Heuristic, binary (paper rule) | 1.36× | 1.27× | 1.12× | 1.28× | 1.27× | 1.11× | 1.24× | 1.20× | 1.11× |
| Heuristic, graded 30/60 m | 1.16× | 1.12× | 1.06× | 1.12× | 1.13× | 1.06× | 1.13× | 1.11× | 1.06× |
| Learned UCB, τ=0.1 | 1.19× | 1.11× | 1.06× | 1.15× | 1.11× | 1.05× | 1.16× | 1.12× | 1.06× |
| Learned UCB, τ=0.2 | 1.23× | 1.15× | 1.07× | 1.15× | 1.12× | 1.05× | 1.20× | 1.15× | 1.08× |
| Learned UCB, τ=0.3 | 1.31× | 1.20× | 1.09× | 1.20× | 1.15× | 1.07× | 1.24× | 1.18× | 1.09× |
| Learned UCB, τ=0.4 | 1.37× | 1.24× | 1.11× | 1.27× | 1.21× | 1.09× | 1.27× | 1.20× | 1.11× |
| Learned UCB, τ=0.5 | 1.38× | 1.25× | 1.11× | 1.30× | 1.23× | 1.10× | 1.28× | 1.21× | 1.11× |
| Learned Thompson, τ=0.3 | 1.35× | 1.23× | 1.11× | 1.27× | 1.20× | 1.09× | 1.27× | 1.20× | 1.11× |
| Oracle (hindsight) | 1.39× | 1.29× | 1.13× | 1.30× | 1.27× | 1.11× | 1.29× | 1.24× | 1.12× |

Measured per-frame cost of always-FULL (model / end-to-end, ms): YOLOv8n 32.0 / 63.0; YOLOv8s 16.4 / 34.8; YOLOv8m 20.9 / 36.4

Projected: profiled forward-pass latency × frames per level (network only). Measured (model): wall-clock per frame for the level each criterion chose, timed on the same decoded frame — pre-processing, forward pass, box decoding and NMS, plus the router's own overhead; learned routers pay the full decode of all levels on their FULL frames. Measured (end-to-end): additionally includes decoding the 4K video frame. All relative to always-FULL measured the same way.
