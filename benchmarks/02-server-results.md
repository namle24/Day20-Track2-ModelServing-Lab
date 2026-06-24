# 02 — llama-server Load Results

Server: local llama.cpp wrapper on `TinyLlama-1.1B Q4_K_M`, CPU backend, `n_threads=8`, `n_ctx=2048`, `n_gpu_layers=0`.

Load-test token caps for CPU-only measurement: `LAB_LOAD_SHORT_MAX_TOKENS=12`, `LAB_LOAD_LONG_MAX_TOKENS=20`.

| Concurrency | Requests | Failures | RPS | E2E P50 (ms) | E2E P95 (ms) | E2E P99 (ms) |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 27 | 0 | 0.46 | 14000 | 30000 | 30000 |
| 50 | 23 | 0 | 0.47 | 30000 | 45000 | 45000 |

## Metrics

`benchmarks/02-server-metrics.csv` recorded `/metrics` after load. Peak observed `llamacpp:kv_cache_usage_ratio` was `0.173828`; `llamacpp:tokens_predicted_total` increased from `1196` to `1252` during the scrape window.

## Observation

This CPU-only laptop saturates on decode before KV cache capacity. Increasing concurrency from 10 to 50 mostly adds queueing latency: RPS stays roughly flat while P50/P95 grow.
