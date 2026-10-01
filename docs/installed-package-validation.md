# Published package validation: 0.1.0

This page preserves the checks of the 0.1.0 wheel. They do not verify the
0.2.0 ONNX default or its installation dependencies.

The public **gemmadecision 0.1.0** wheel passed functional checks in fresh
Modal CPU and GPU environments on September 27, 2026. Both environments
installed `gemmadecision==0.1.0` from PyPI. No repository package code or local
model files were mounted; only the test harness was supplied.

| Environment | Local `decide()` / `rank()` | Native PydanticAI locally | Granian HTTP + SDK + native PydanticAI |
|---|---|---|---|
| 4 CPU cores, 8 GiB RAM | Passed | Passed | Passed |
| NVIDIA H100 80 GB, 2 CPU cores, 16 GiB RAM | Passed | Passed | Passed |

The CPU image first installed PyTorch `2.14.0+cpu` from the official PyTorch
CPU index. The GPU image used the package's normal dependency resolution:
PyTorch `2.14.0` with CUDA 13.0. Both used Python 3.11.12, Transformers 5.17.0,
PydanticAI 2.51.0, Granian 2.8.3 and FastAPI 0.136.3.

The checks verified installation under `site-packages`, public wheel origin,
and the immutable Hugging Face model revision
`785d530221c990671f29976902540101bb9c7647`. Each environment downloaded the model
without a Hugging Face token and checked the encoder, decision head and
configuration hashes. Local calls reused one loaded engine. That process
exited before the real Granian server started, preventing two model copies
from occupying memory together. The HTTP check made one inference request
through `AsyncDecisionClient` and native PydanticAI, returning a typed
Literal/bool result.

Both installations used this [published wheel](https://files.pythonhosted.org/packages/12/9f/359c665e94b65ff9d7c3c69f6c9761d02f8de59497c10a4831748af6087b/gemmadecision-0.1.0-py3-none-any.whl):

```text
SHA256 5897050c6e54deca3a5a683fa8a7a30ae2e2d1b496ae60e3362edca17d3cef52
```

These are installation and integration checks using short, two-option examples.
They do not measure accuracy, steady-state latency, throughput or CPU-versus-GPU
performance. The single-call timings retained in the reports include different
initialization effects and should not be used for those comparisons.

Evidence: [CPU report](measurements/pypi-0.1.0-cpu.json),
[GPU report](measurements/pypi-0.1.0-gpu.json), and
[provenance and file hashes](measurements/PYPI_VALIDATION_PROVENANCE.json).
Only the temporary Hugging Face cache directory was replaced with `$HF_HOME`
in the published reports; the measured values are unchanged.
