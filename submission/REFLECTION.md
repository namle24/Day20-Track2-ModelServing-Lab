# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** Nam Le
**Cohort:** AICB-P2T2 Track 2
**Ngày submit:** 2026-06-24

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

> Paste output của `python 00-setup/detect-hardware.py` vào đây, hoặc điền thủ công:

- **OS:** Linux
- **CPU:** Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
- **Cores:** 8 physical / 8 logical
- **CPU extensions:** AVX2, no AVX-512
- **RAM:** 7.6 GB
- **Accelerator:** CPU only
- **llama.cpp backend đã chọn:** CPU
- **Recommended model tier:** TinyLlama-1.1B (Q4_K_M)

**Setup story** (≤ 80 chữ): những gì cần thay đổi để lab chạy được trên máy bạn (vd: dùng WSL2, install CUDA Toolkit, fall back sang Vulkan vì ROCm phiên bản kén, tắt antivirus để pip install nhanh hơn, v.v.):

Setup chạy theo CPU-only path. Điểm cần sửa là server dependency và metrics endpoint: môi trường `llama_cpp.server` hiện tại thiếu server extras và không expose `/metrics`, nên lab dùng wrapper FastAPI mỏng quanh `llama_cpp.Llama` để giữ `/v1/chat/completions` và thêm Prometheus metrics.

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

> Paste bảng từ `benchmarks/01-quickstart-results.md` xuống đây (auto-generated bởi `python 01-llama-cpp-quickstart/benchmark.py`).

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 837 | 292 / 345 | 51.2 / 91.0 | 3458 / 3895 / 4025 | 19.5 |
| tinyllama-1.1b-chat-v1.0.Q2_K.gguf | 448 | 366 / 804 | 56.1 / 66.2 | 3543 / 4487 / 4597 | 17.8 |

**Một quan sát** (≤ 50 chữ): Q4_K_M vs Q2_K trên máy bạn — số liệu nói gì? Quality đáng đánh đổi không?

Trên máy này Q4_K_M nhanh hơn Q2_K ở decode P50 và ổn định hơn ở TTFT P95, dù load chậm hơn. Vì RAM đủ cho TinyLlama Q4, tôi chọn Q4_K_M.

---

## 3. Track 02 — llama-server load test

> Chạy 2 lần locust ở concurrency 10 và 50, paste tóm tắt bên dưới.

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.46 | 14000 | 30000 | 30000 | 0 |
| 50 | 0.47 | 30000 | 45000 | 45000 | 0 |

**Batching observation** (từ `record-metrics.py`): peak `llamacpp:n_busy_slots_per_decode` / `requests_processing` ở concurrency 50 = _<…>_, nghĩa là …
**KV-cache observation** (từ `record-metrics.py`): peak `llamacpp:kv_cache_usage_ratio` ở concurrency 50 = 0.173828, nghĩa là KV cache chưa phải bottleneck chính. Bottleneck là decode CPU và queueing: tăng concurrency từ 10 lên 50 không tăng RPS đáng kể nhưng làm P50/P95 tăng.

Load test dùng `LAB_LOAD_SHORT_MAX_TOKENS=12` và `LAB_LOAD_LONG_MAX_TOKENS=20` để đo được goodput trên CPU-only laptop trong cửa sổ 60 giây.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub: localhost only, no cloud deploy for this personal laptop run
- **N17 (Data pipeline):** stub: static in-memory document set
- **N18 (Lakehouse):** stub: no Delta/Iceberg table in this repo run
- **N19 (Vector + Feature Store):** stub: `TOY_DOCS` keyword retrieval with provenance ids

**Nơi tốn nhiều ms nhất** trong pipeline (đo bằng `time.perf_counter` trong `pipeline.py`):

- embed: 0.0 ms (stubbed, no embedding model)
- retrieve: 0.0 ms
- llama-server: 2617.3-6963.3 ms across 3 queries

**Reflection** (≤ 60 chữ): bottleneck nằm ở đâu? Có khớp với kỳ vọng không?

The bottleneck is clearly llama-server inference, not retrieval. That matches expectation for a tiny in-memory retriever on CPU-only hardware.

---

## 5. Bonus — The single change that mattered most

> **Most important section.** Pick **một** thay đổi từ bonus track (build flag, thread sweep, quant pick, GPU offload, KV-cache quantization, speculative decoding, bất cứ challenge nào trong `BONUS-llama-cpp-optimization/CHALLENGES.md`) đã tạo ra speedup lớn nhất trên máy bạn.

**Change:** cap generated tokens for CPU-only serving load tests and pipeline (`LAB_LOAD_SHORT_MAX_TOKENS=12`, `LAB_LOAD_LONG_MAX_TOKENS=20`, `LAB_PIPELINE_MAX_TOKENS=32`).

**Before vs after** (paste 2-3 dòng từ sweep output):

```
before: -u 50, max_tokens 80/160 -> 0 completed requests in 60s, no usable P95
after:  -u 50, max_tokens 12/20 -> 23 completed requests in 60s, P95 45s, 0 failures
speedup: measurable goodput changed from 0.00 to 0.47 req/s for the 60s window
```

**Tại sao nó work** (1–2 đoạn ngắn — đây là phần grader đọc kỹ nhất):

Trên CPU-only laptop này, decode là phần đắt nhất vì mỗi output token phải đi qua model weights và bị giới hạn bởi CPU/memory bandwidth. Khi concurrency cao, các request xếp hàng sau cùng một model instance; nếu mỗi request sinh 80-160 token, cửa sổ 60 giây không kịp hoàn tất request nào ở `-u 50`.

Giới hạn output token không làm model "nhanh hơn" theo tok/s, nhưng làm workload khớp SLO hơn: ít decode token hơn, queue ngắn hơn, và goodput trong 60 giây trở nên đo được. Đây là bài học production quan trọng: với phần cứng yếu, SLO phải đi cùng giới hạn response length, không chỉ tăng concurrency.

---

## 6. (Optional) Điều ngạc nhiên nhất

_(1–2 câu — không bắt buộc, nhưng người grader đọc tất cả)_

Điều ngạc nhiên nhất là Q4_K_M không chậm hơn Q2_K trên benchmark ngắn; trên máy này Q4_K_M có TPOT P50 tốt hơn và đáng dùng hơn.

---

## 7. Self-graded checklist

- [x] `hardware.json` đã commit
- [x] `models/active.json` đã commit (hoặc paste path snapshot vào section 1)
- [x] `benchmarks/01-quickstart-results.md` đã commit
- [x] `benchmarks/02-server-results.md` (hoặc CSV từ `record-metrics.py`) đã commit
- [ ] `benchmarks/bonus-*.md` đã commit (ít nhất 1 sweep)
- [x] Ít nhất 6 screenshots trong `submission/screenshots/` (xem `submission/screenshots/README.md`)
- [x] `make verify` exit 0 (chạy ngay trước khi push)
- [ ] Repo trên GitHub ở chế độ **public**
- [ ] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.
