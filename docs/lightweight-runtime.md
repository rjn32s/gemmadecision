# Lightweight CPU runtime: 0.2.0

Version 0.2.0 runs local decisions with **ONNX Runtime on CPU**, without
installing Torch or Transformers. The same `decide()` and `rank()` calls work.
HTTP serving, PydanticAI, native Torch, and vLLM are optional extras.

In the CPU validation below, two short choices took **110 ms median**. ONNX
also preserved the Torch model's top choice on all **700 development cases**.
These results concern the stated workloads and model conversion; they do not
establish a universal speedup or accuracy on a new task.

## Install or upgrade

```bash
python -m pip install --upgrade gemmadecision==0.2.0
```

In a notebook, run `%pip install --upgrade gemmadecision==0.2.0` in a cell,
then **restart the runtime/kernel** before running the code below. Installing
the upgrade alone does not replace already imported 0.1.0 modules in the
running Python process.

```python
from gemmadecision import decide

print(decide("I was charged twice", choices=["billing", "technical"]))
```

| Add this capability | Install |
|---|---|
| Local CPU decisions | `pip install gemmadecision` |
| Granian Rust HTTP server | `pip install 'gemmadecision[serve]'` |
| Native PydanticAI agents | `pip install 'gemmadecision[pydantic-ai]'` |
| Native Torch, CUDA, or Apple MPS | `pip install 'gemmadecision[torch]'` |
| vLLM on supported Linux/CUDA | `pip install 'gemmadecision[vllm]'` |

Extras combine, for example `gemmadecision[serve,pydantic-ai]`. The default
`auto` path uses CPU ONNX. Explicit `device="cuda"` or `device="mps"` selects
Torch and requires its extra. See [hardware settings](reference/python.md#decisionengine)
and [serving](guides/serving.md).

## Why the notebook dependency failure happened

The 0.1.0 package required Torch 2.13 or newer. Installing it into a notebook
with Torch 2.11 and a matching torchvision 0.26 could upgrade Torch while
leaving the existing torchvision binary tied to 2.11. The resulting
`operator torchvision::nms does not exist` error can surface as a failure to
import `Gemma3TextModel`: Transformers' model-loading path also checks installed
optional vision/audio libraries, even for this text model.

The 0.2.0 default avoids that dependency chain. It does not replace or import
Torch, torchvision, or torchaudio. It does not repair other applications that
already have incompatible Torch packages. Explicit native Torch users should
keep that environment's Torch/vision/audio builds compatible; the
[notebook recovery guide](guides/troubleshooting.md#upgrade-a-010-notebook)
also covers old local model-directory settings.

## What the runtime artifact contains

The graph includes the tuned encoder, normalized last-token pooling, and the
trained scalar head. It returns the same kind of raw ranking score. The API
still applies the existing softmax temperature, **4.136820402388508**, and
retains the choice, yes/no, and ordered-score response semantics.

Weights that round-trip exactly are stored in FP16 or BF16 and restored to
their original **FP32 values before computation**. Other values remain FP32.
The selected format uses lossless storage conversion; it is not INT8
quantization or FP16 model computation. Different runtime kernels can still
produce small floating-point differences, which are measured below.

The graph is **538,514,811 bytes**. Its complete runtime files total
**573,083,893 bytes**: approximately **573 MB / 546.5 MiB**, including the
tokenizer and configuration. This is a one-time cached model download,
separate from the Python dependencies. Later calls in the same process reuse
the loaded model.

The compact file avoids the size of a naive FP32 ONNX export. The original
checkpoint already used BF16 storage and was about the same size; this is
**not a 50% reduction against that checkpoint**. FP32 inference still needs
memory for restored weights, activations, and runtime workspaces. Download
size is not a RAM requirement. The validation process peaked at about
**2.72 GiB RSS**, but it ran Torch and ONNX sequentially in one process, so
that number is not an isolated ONNX memory measurement or minimum.

The package verifies the pinned export manifest and every listed inference
file. The source weights remain model revision
`785d530221c990671f29976902540101bb9c7647`; conversion does not retrain them.

## Public PyPI installation

After publication, a fresh Python 3.13.3 environment on Modal, with four CPU
cores and 8 GiB RAM, installed **`gemmadecision==0.2.0` directly from public
PyPI**. The exact example above returned `billing`. No Torch, torchvision,
torchaudio, or Transformers packages were installed or imported, and both
`decide()` and `rank()` worked again in a new process using the model cache
offline.

| Measurement | Result |
|---|---:|
| Ordinary pip install, including dependency resolution | 7.5920 s |
| Compressed package wheels, including dependencies | 54,019,036 bytes (~54 MB) |
| Published GemmaDecision wheel within that total | 47,631 bytes |
| First exact example, including model download and loading | 6.669 s |
| Model loading in a new process from the cache, offline | 4.090 s |
| Warm `decide()` median for two descriptive choices, 34/35 joint tokens | 83.49 ms |

The **573 MB model download is separate** from the package archives. Install
time excludes virtual-environment creation and model download. The warm
median covers ten calls after three warmups, using the descriptive two-choice
fixture recorded in the report rather than the short first-call example.
These are functional and latency observations from one run, not accuracy
measurements or a speed comparison with the separate hosts used below.

The [public installation report](measurements/pypi-0.2.0-minimal.json) records
the public package URL, dependency versions, exact samples, and verified model
pins. It retains the shared harness's candidate-prefixed measurement keys;
`installation_source: public_pypi` and the artifact URL identify this run.
The published wheel SHA256 is
`3b72a2bd3f0831ef623a96d5f9580c7829cbc0bae48a7af93d937029b9afae1d`.
Only two temporary artifact-path fields were removed from the report copy,
with its original SHA256 recorded in the provenance entry.
All **18 runtime Python files** in the published wheel were also verified
byte-for-byte against the candidate wheel and committed source; see the
[wheel/source comparison](measurements/pypi-0.2.0-source-verification.json).

## Fresh installation of the candidate wheel

A separate smoke test installed the **0.2.0 candidate wheel** into a fresh
Python 3.13.3 environment on Modal, with four CPU cores and 8 GiB RAM. This
checked the built distribution before publication; it was **not an install
of 0.2.0 from public PyPI**.

| Measurement | Result |
|---|---:|
| Ordinary pip install, including dependency resolution | 10.4266 s |
| Compressed package wheels, including dependencies | 54,018,997 bytes (~54 MB) |
| GemmaDecision candidate wheel within that total | 47,592 bytes |
| First `decide("I was charged twice", choices=["billing", "technical"])`, including model download and loading | 8.448 s |
| Model loading in a new process from the cache, offline | 5.359 s |
| Warm `decide()` median for two descriptive choices, 34/35 joint tokens | 142.75 ms |

The model files add a separate **573 MB** one-time download. Package archive
sizes exclude pip metadata, network overhead, and extracted installation
size. Installation time excludes virtual-environment creation and model
download. The first decision returned `billing`; that single example checks
that inference works and is not an accuracy evaluation.

Torch, torchvision, torchaudio, and Transformers were absent from the fresh
installation, and the inference checks made no forbidden framework import
attempts. Both `decide()` and `rank()` also worked in a second process with
network access to the model disabled and the previously downloaded cache.

The warm measurement used ten timed calls after three warmups, with the
model already loaded. Its prompt and host differ from the same-host backend
comparison below, so the two tables should not be used to infer a speed
change.

The [candidate installation report](measurements/candidate-0.2.0-minimal.json)
records dependency versions, wheel hashes, every timing sample, and model
pins. Only two remote artifact-path fields were omitted from the copied
report; a provenance entry records the original report hash. The candidate
wheel SHA256 was
`0d96bcfb0e852caeb8673b0ed7b9afca0612a1213f6d13ce8a4a86edcbd45341`.
It downloaded the verified export at Hub revision
`254ac03b1ec6c96af3b9e605bd9c6f5cc0156fae`.

Two additional candidate-wheel checks passed. A
[notebook compatibility simulation](measurements/candidate-0.2.0-colab.json)
kept existing **Torch 2.11.0+cpu and torchvision 0.26.0+cpu unchanged** and
returned `billing` for the same user example without importing either
framework, even with a sentinel that rejects torchvision imports. This
simulated that dependency setup; it did not run inside hosted Google Colab.
The [optional integrations check](measurements/candidate-0.2.0-integrations.json)
installed `[serve,pydantic-ai]` and passed a local native PydanticAI agent,
an actual Granian HTTP server, unauthorized-request rejection, an authenticated
SDK decision, and a remote native PydanticAI agent, all using CPU ONNX without
Torch. These are functional checks, not additional accuracy measurements.

## Conversion fidelity on development data

The check used the existing, frozen **700-example development set**, with
**2,300 candidate pairs**. All 700 examples fit the input limits; none were
skipped. Their joint candidate inputs ranged from 25 to 141 tokens. No final
test or calibration data was opened, and neither weights nor temperature were
fitted during this validation.

| Development task family | Cases | Same top choice as CPU Torch FP32 |
|---|---:|---:|
| Intent routing | 300 | 300 |
| Evidence relation | 300 | 300 |
| Rule compliance | 100 | 100 |
| **Total** | **700** | **700 / 700 (100%)** |

Across the candidate scores, maximum absolute error was **0.00017345** and
mean absolute error was **0.00000812**. Maximum absolute derived-probability
error was **0.00000908**.

This measures conversion fidelity on development data, **not a new blind
accuracy benchmark**. Matching the reference does not make its decisions
correct, nor guarantee agreement for every future input or near tie. Existing
model evaluations remain in the [model card](https://huggingface.co/rajan2k/GemmaDecision-270M).

## CPU latency on the same host

Both backends ran sequentially on the same remote Linux host with four
available CPU cores and four inference threads. The CPU model was reported
as `unknown`. The environment used Python 3.13.3, Torch 2.14.0+cpu,
Transformers 5.17.0, and ONNX Runtime 1.30.0.

Each measurement covers a complete `DecisionEngine.rank()` call, including
validation, tokenization, scoring, and response construction. The model was
already loaded, the score cache was off, and there were three warmups per
workload. HTTP, download, and model initialization are excluded. Both used
a 4,096-padded-token budget and a maximum batch size of 16.

| Joint tokens per candidate | Choices | Torch FP32 median | ONNX median | Timed calls per backend |
|---:|---:|---:|---:|---:|
| 29 | 2 | 149.2 ms | **110.0 ms** | 10 |
| 29 | 4 | 186.3 ms | 211.5 ms | 10 |
| 128 | 2 | 417.2 ms | 321.2 ms | 10 |
| 128 | 4 | 683.9 ms | 780.7 ms | 10 |
| 512 | 2 | 1,262.3 ms | 1,137.5 ms | 5 |
| 512 | 4 | 2,457.2 ms | 2,387.0 ms | 5 |

ONNX was slower for the four-choice 29-token and 128-token workloads in this
run. These small repeated samples support an environment-specific comparison,
not a general latency or throughput guarantee. The 110 ms result applies to
two short choices, not every decision. The primary installation benefit is
avoiding the Torch/Transformers dependency stack for CPU inference.

## Evidence and reproduction

[Raw aggregate validation report](measurements/onnx-0.2.0-validation.json)
contains every timing sample, task counts, environment, script hash, and export
manifest hash. The report is copied unchanged. Its `onnx_revision: null` records
that this check used a verified prepublication artifact, before an immutable
Hub export revision was assigned.

```text
Validation report SHA256
bc4851eca09d1f5ee2c40906f0e0cb9b667f79eac6e9a3c84b6c635b995b7566

Validated export manifest SHA256
f541a308c7cc4c75b645d2a026bde4771c0877e7129147cb260c1b8e305be178
```

The source includes the [exporter](https://github.com/rjn32s/gemmadecision/blob/main/scripts/export_onnx.py),
[lossless-storage conversion](https://github.com/rjn32s/gemmadecision/blob/main/scripts/optimize_onnx_candidates.py),
and [validation harness](https://github.com/rjn32s/gemmadecision/blob/main/scripts/validate_onnx_release.py).
These are remote validation tools, not steps needed for ordinary inference.
The separate [0.1.0 CPU report](cpu-latency.md) remains historical and uses
different measurements.
