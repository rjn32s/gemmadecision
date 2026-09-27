# CPU inference latency

The published **gemmadecision 0.1.0** wheel was measured on Modal with **4 CPU
cores and 8 GiB reserved RAM**, CPU-only PyTorch 2.14.0, FP32 model weights,
four Torch threads and one inter-op thread. The CPU model was not exposed by
the container (`unknown`); these results should not be treated as timings for
every laptop or processor.

Each timing covers one complete `rank()` call: input validation, tokenization,
all candidate scoring and response construction. There is no HTTP in these
measurements. The model stays loaded, the score cache is disabled, and each
workload receives three warmup calls. Tokens below count the complete joint
state, question and one candidate, including special tokens.

| Tokens per candidate input | Choices | Median latency | Sample p95 | Timed calls |
|---:|---:|---:|---:|---:|
| 30 | 2 | 124 ms | 162 ms | 10 |
| 30 | 4 | 189 ms | 216 ms | 10 |
| 128 | 2 | 397 ms | 430 ms | 10 |
| 128 | 4 | 698 ms | 734 ms | 10 |
| 512 | 2 | 1365 ms | 1414 ms | 5 |

These small samples show how latency grows with input length and candidate
count; p95 from 5–10 calls is not a production tail-latency guarantee. Inputs
are fixed synthetic fixtures, and no decision accuracy was measured. This
CPU experiment is not an apples-to-apples comparison with the separate GPU
fixtures. The earlier 71 ms CPU number was one shorter installation smoke-test
call, not the median for these workloads.

In this run, an anonymous fresh model download took **2.95 s**,
model loading plus integrity checks took **9.21 s**,
and the first ranking after loading took **124 ms**.
That is **12.28 s** for download, load and first
ranking, excluding **3.82 s** of imports and
distribution checks. Network and cold-start time depend on the environment.
Peak process RSS was **2.54 GiB**. An application
reusing the default engine pays the model-loading cost once per process.

The function ran for 46.52 s. Its rate-based CPU/RAM compute
estimate was **$0.00326**, excluding image builds,
startup, storage and transfer.

[Raw samples and environment](measurements/pypi-0.1.0-cpu-latency.json) ·
[Compute record](measurements/pypi-0.1.0-cpu-latency-compute.json) ·
[Exact measurement harness](measurements/pypi-0.1.0-cpu-latency-harness.py).
The report records the harness SHA256 and the public wheel's artifact hash.
The harness expects a fresh Linux CPU environment with the pinned package
installed using `pip --report /tmp/gemmadecision-install.json`, a fresh `HF_HOME`,
and a JSON output path as its argument. It downloads the immutable public model
without a token and verifies all inference files before measuring.
