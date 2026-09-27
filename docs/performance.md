# Measured serving performance

On one **NVIDIA H100 80GB HBM3**, scoring 32 short joint inputs took **27.59 ms**
with batching, compared with **753.36 ms** in the frozen published reference and
**758.77 ms** in this package's strict singleton mode. That is **27.31×** faster
than the reference for this specific workload. Four short candidates took
**25.41 ms**, compared with **94.35 ms** in the reference.

These are local backend timings, including tokenization and the scalar head,
with explicit GPU synchronization. They exclude HTTP, input validation, and
downloads. Each workload had three warmups and only **five measured repeats**;
these results establish a useful initial measurement, not a production SLA or
model-accuracy result. Full samples and synthetic inputs are in
[h100-backend.json](measurements/h100-backend.json).

## Device and model

Measurements used the fully tuned GemmaDecision v0.4.0 encoder and scalar head,
model revision `785d530221c990671f29976902540101bb9c7647`, with verified file
hashes. The CUDA encoder used BF16; last-token normalization and the learned
scalar head used FP32. The environment was Python 3.11.12, Torch 2.14.0,
Transformers 5.17.0, CUDA 13.0, with two Torch CPU threads and TF32 matrix
multiplication disabled. Source was loaded directly from the recorded source
snapshot; the absent installed package version in the JSON is intentional.

The batch limit was 32 inputs and 8,192 **padded** tokens. The cache was off.
The reference scores one state/candidate pair at a time. This package's strict
mode preserves that numerical batch shape. Its normal mode buckets inputs by
length, pads on the right, scores bounded batches, and restores original order.

## Backend latency

All values below are medians in milliseconds. A pair means one complete joint
state/question/candidate input, not one HTTP request. Long workloads can require
several forward passes because the padded-token budget still applies.

| Workload | Tokens per pair | Frozen reference | Package strict | Package batched | Speedup vs reference |
|---|---:|---:|---:|---:|---:|
| Short, 1 pair | 49 | 23.35 | 23.63 | 23.64 | 0.99× |
| Short, 4 pairs | 49–50 | 94.35 | 94.24 | 25.41 | 3.71× |
| Short, 16 pairs | 49–52 | 373.87 | 383.23 | 26.01 | 14.37× |
| Short, 32 pairs | 49–55 | 753.36 | 758.77 | 27.59 | 27.31× |
| Long, 1 pair | 721 | 25.26 | 25.29 | 25.20 | 1.00× |
| Long, 4 pairs | 721–722 | 99.40 | 100.64 | 28.68 | 3.47× |
| Long, 16 pairs | 721–724 | 398.57 | 400.24 | 62.71 | 6.36× |
| Long, 32 pairs | 721–727 | 792.84 | 806.43 | 100.67 | 7.88× |

For the 32 short pairs, batched throughput was **1,160 pairs/s**. This is not
HTTP requests/s. The backend reported peak Torch-allocator memory of
**726,264,320 bytes allocated** and **838,860,800 bytes reserved**. Those numbers
exclude driver and non-Torch allocations and do not establish a minimum GPU
VRAM requirement for all possible inputs.

## Numerical agreement

On **12 hand-authored cases containing 96 candidate pairs**, strict mode matched
the reference scores exactly. Batched mode selected the same top candidate in
all 12 cases; maximum absolute score difference was **0.0000152588**, and maximum
probability difference was **0.000000455138** using the fixed external temperature
`4.136820402388508`. The smallest observed reference top-two margin was about
0.0136. This fixture check cannot rule out different results on closer ties,
other kernels, devices, or model versions.

The inputs are unlabeled serving fixtures. Agreement with the reference is
**not decision accuracy**, JevBench performance, or a new calibration result.
The experiment did not train, quantize, or change the model.

## Real Rust HTTP server

The second run started the actual CLI and **Granian 2.8.3 Rust HTTP runtime**, then
used the Python SDK over loopback HTTP. It used the same H100 class, model,
precision, batch limits, a 1 ms batching window, and no score cache. Each request
had one choice question with four candidates and a distinct case ID, so identical
input deduplication did not inflate throughput. After three warmup requests,
**32 requests per concurrency level** were measured:

| Concurrent clients | HTTP requests/s | Candidate pairs/s | Request p50 | Request p95 | Scheduler batches |
|---|---:|---:|---:|---:|---:|
| 1 | 26.99 | 107.96 | 25.63 ms | 70.17 ms | 32 |
| 8 | 93.60 | 374.41 | 81.29 ms | 121.53 ms | 8 |
| 32 | 85.59 | 342.37 | 361.92 ms | 366.58 ms | 2 |

Request latency starts after the client concurrency semaphore is acquired;
it includes server queueing, validation, tokenization, inference, and HTTP,
but excludes waiting for a client slot. Throughput uses the elapsed time to
complete each 32-request group. Scheduler batches count grouped engine calls;
each can contain several bounded encoder forwards. Concurrency 32 increased
latency and did not improve throughput over 8 in this small experiment.

With only 32 requests per level, the percentiles are noisy and do not describe
steady-state production tail latency. There is no cross-machine network in this
measurement. See [h100-http.json](measurements/h100-http.json) for all results.
Rust handles HTTP connections; the measured transformer math remains in
Torch/SDPA. These results do not isolate a Rust-versus-Python HTTP speedup.

## Real API checks

The successful HTTP run also verified all three question types (`choice`,
`noul`, and `score`), the ranking endpoint, a native PydanticAI model returning
a typed Literal/bool result, missing-key rejection (401), and an overlong state
rejection (422). Readiness took **12.75 s** using an existing local model
directory; the first inference still needed additional warmup.

After shutting down the HTTP model process, a fresh process verified the simple
`decide()` and `rank()` APIs with the pinned local files and network access to
Hugging Face disabled. The first `decide()` including model load took **9.88 s**;
one subsequent call took **23.35 ms**. These single calls are smoke checks, not
latency distributions or uncached download timings.

The same process then ran a real PydanticAI
`Agent(GemmaDecisionModel.local(), output_type=Literal["billing", "technical"])`.
It returned the valid Literal `billing`, reported `local://gemmadecision`, and
reused the same process-local engine. The warm Agent call took **154.12 ms**.
No external LLM or HTTP server was used for that local Agent check. Correct
output types and engine reuse were tested; application accuracy was not scored.

## Reproduction and evidence

Run the GPU scripts on a CUDA host with the package installed and a complete
pinned model directory:

```sh
python scripts/benchmark_backends.py --help
python scripts/benchmark_server.py --model-path MODEL_DIR --output server-results.json
```

The backend report includes every fixture, five timing samples per workload,
per-case parity, and environment metadata. The corresponding
[backend source hashes](measurements/h100-backend-source-hashes.json) and
[HTTP source hashes](measurements/h100-http-source-hashes.json) identify the
exact code used. [Provenance](measurements/PROVENANCE.json) records original and
published JSON hashes; the HTTP report normalizes interpreter/model-directory
paths for portability without changing measurements.

Estimated function compute was **$0.09613** for the backend run and **$0.04331**
for the successful HTTP/API run: **$0.13943 combined**, excluding startup,
storage, and transfer. These are rate-based estimates, not a billing receipt;
see [backend compute](measurements/h100-backend-compute.json) and
[HTTP compute](measurements/h100-http-compute.json). The first combined run's
HTTP startup failed on a Granian keyword mismatch; its compute record preserves
that failed status. The later successful HTTP run used the corrected CLI. A
regression test now constructs the real Granian object while replacing only
its `serve()` method, so constructor errors are caught without loading weights.

The preceding measurements cover the Torch backend. They establish no CPU/Mac
latency claims. The separately measured vLLM backend is reported below.
Validate the intended deployment hardware and workload before
choosing concurrency and batch limits.

## Input and batching contract

State/question and candidate limits remain 2,048 and 768 tokens respectively,
including special tokens. The encoder validates complete joint inputs and never
truncates text. A single pair longer than `max_batch_tokens` is rejected instead
of exceeding that configured batch budget. Candidate and state cannot be cached
as independent embeddings because this model encodes their concatenation
jointly. Increasing an architectural context limit is not equivalent to
validating a new serving input contract.

## vLLM validation on H100

The optional **vLLM 0.30.0 backend passed real GPU and HTTP/API checks** on an
H100 80GB HBM3. Batched PyTorch was faster for every backend workload in this
experiment, so PyTorch remains the default. This comparison uses the frozen
reference and both backends measured sequentially in **the same run with
Torch 2.13.0+cu130**, Python 3.12.3, Transformers 5.17.0, CUDA 13.0 and two CPU
threads. It does not compare vLLM against the earlier Torch 2.14.0 run.

The model revision and 96 synthetic pairs were unchanged. Prefix and score
caches were disabled; vLLM used LAST pooling, BF16 CUDA inference, eager
execution, a 3,072-token joint limit and a CPU FP32 normalization/head. The
backend timings below include tokenization and the head, exclude HTTP, and use
three warmups plus five measured repetitions. PyTorch's 8,192 padded-token
budget can split long workloads into multiple forwards; vLLM schedules its own
input batches. These are the tested default configurations, not a search for
each runtime's best tuning.

| Workload | Frozen reference p50 | Batched Torch p50 | vLLM p50 | vLLM p95 |
|---|---:|---:|---:|---:|
| Short, 1 pair | 14.94 ms | 14.97 ms | 20.57 ms | 21.24 ms |
| Short, 4 pairs | 59.85 ms | 16.64 ms | 43.03 ms | 43.31 ms |
| Short, 16 pairs | 236.79 ms | 17.53 ms | 49.59 ms | 49.93 ms |
| Short, 32 pairs | 474.80 ms | 18.93 ms | 56.85 ms | 59.69 ms |
| Long, 1 pair | 16.23 ms | 16.76 ms | 21.90 ms | 22.74 ms |
| Long, 4 pairs | 64.57 ms | 19.58 ms | 46.12 ms | 54.69 ms |
| Long, 16 pairs | 257.96 ms | 51.74 ms | 63.59 ms | 63.89 ms |
| Long, 32 pairs | 516.84 ms | 91.53 ms | 115.11 ms | 119.24 ms |

For 32 short pairs, throughput was **562.90 pairs/s with vLLM**, versus
**1,690.71 pairs/s with batched Torch** in that run. vLLM backend initialization
took **68.56 s** and its first request another **39.91 ms**. Initialization
includes engine setup with local weights, not downloading the model. The
measurement script cannot report vLLM worker memory through the parent
process's Torch allocator, so those fields are explicitly unavailable.
See [vllm-h100-backend.json](measurements/vllm-h100-backend.json).

### Numerical agreement, not accuracy

The same **12 cases / 96 candidate pairs** gave **12/12 top-candidate
agreement** between vLLM and the frozen reference. Maximum absolute score
difference was **0.3179318905**, mean absolute score difference **0.0661979032**,
and maximum probability difference **0.0087380903** using the unchanged
temperature `4.136820402388508`. Thus the measured outputs are not numerically
equivalent, even though these cases retained their top choice. Batched Torch
in the same run had maximum score difference **0.0000152588** and also agreed
on all 12 top choices; strict Torch matched the reference exactly.

This is an unlabeled serving fixture check, not accuracy, a JevBench rerun or
evidence that every close decision is preserved. Use strict Torch when
matching the singleton reference's numerical behavior matters.

### Real vLLM HTTP serving

The actual `gemmadecision serve --backend vllm` command ran behind Granian
2.8.3. After three warmup requests, each concurrency group contained 32
distinct requests with four candidates each, with caching off and a 1 ms
microbatch window:

| Concurrent clients | HTTP requests/s | Candidate pairs/s | Request p50 | Request p95 |
|---|---:|---:|---:|---:|
| 1 | 21.12 | 84.49 | 45.65 ms | 56.11 ms |
| 8 | 70.11 | 280.45 | 113.98 ms | 124.41 ms |
| 32 | 111.76 | 447.05 | 276.59 ms | 282.13 ms |

These are loopback HTTP SDK measurements; client-semaphore waiting is excluded
from latency, while server queueing is included. The small sample counts do
not establish production tail latency. No claim is made that cross-run HTTP
differences from the Torch 2.14.0 experiment are attributable to vLLM alone.
Startup to readiness was **62.42 s** with existing local model files.

The run verified choice/noul/score responses, the ranking route, native
PydanticAI Literal/bool output through the HTTP client, missing-key rejection
(401), and oversized-state rejection (422). After the vLLM server stopped,
the script also checked the simple `decide()` / `rank()` and local PydanticAI
APIs. **Those trailing `simple_api` checks used the default Torch backend**,
as their recorded engine metadata shows; they are not vLLM measurements.
See [vllm-h100-http.json](measurements/vllm-h100-http.json).

### vLLM reproducibility and runtime notes

```sh
python scripts/benchmark_backends.py --model-path MODEL_DIR --backend vllm --output backend-results.json
python scripts/benchmark_server.py --model-path MODEL_DIR --backend vllm --output server-results.json
```

The exact [source hashes](measurements/vllm-source-hashes.json),
[compute record](measurements/vllm-compute.json) and
[portable-file provenance](measurements/PROVENANCE.json) are published. Backend
and HTTP checks completed successfully in one **230.26-second function**, with
estimated function compute **$0.27498**, including the reference/Torch
comparisons and the subsequent API checks. The estimate excludes startup,
storage and transfer; it is not a billing receipt.

The validation used the official vLLM Docker image with a launcher-pinned
NumPy 2.4.6. That override conflicted with requirements of the image's unused
`mistral-common` and `lmcache` packages. The tested Gemma pooling/API path
completed, but those extra components were not validated. An independent,
clean Linux/Python 3.12 resolution of `gemmadecision[vllm]` succeeded with
NumPy **2.3.5** and no `lmcache` dependency. Let a fresh environment resolve
dependencies; do not reproduce the image's forced NumPy override. The exact
fresh installation was not separately GPU-tested. Details are in
[runtime notes](measurements/vllm-runtime-notes.json).
