# Python API

These APIs are available in `gemmadecision==0.2.0`. Importing the package does
not load model weights. The simple local functions load and reuse one engine on
first use; HTTP clients connect to a separately running server.

## Simple local calls

```python
from gemmadecision import decide, rank


def main():
    teams = {
        "billing": "Help with payments, invoices, and duplicate charges.",
        "technical": "Help with product errors and broken features.",
    }
    state = "I was charged twice for the same order."
    print(decide(state, teams))
    print(rank(state, teams).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

| Function | Return | Behavior |
|---|---|---|
| `decide(state, choices, *, question="Choose the best option.")` | `str` | Selects a label from a mapping, or the selected text from a list/tuple of distinct strings. |
| `rank(state, choices, *, question="")` | `RankingResponse` | Returns every candidate in descending score order. Accepts a mapping or list of strings. |

For a mapping, keys are application labels and values describe the choices.
For a list passed to `rank`, response labels are string indices (`"0"`, `"1"`,
…). `ranked[i].text` retains the original candidate text. Exact-score ties keep
input order. Supply 2–64 distinct, nonempty candidate descriptions.

`state` accepts JSON-compatible text, objects, arrays, numbers, booleans, or
`None`; the rendered state/question must be nonempty. JSON-object and JSON-array
strings are parsed into structured state. There is no generated explanation.

Set `GEMMADECISION_MODEL_PATH` before the first simple call to use an existing
model directory. For explicit device or backend control, create an engine.

## `DecisionEngine`

```python
from gemmadecision import Choice, DecisionEngine, Noul, Score


def main():
    engine = DecisionEngine.from_pretrained(device="cpu")
    result = engine.system_one("A customer reports a duplicate charge today.", {
        "team": Choice(
            instructions="Choose the support team.",
            criteria={"billing": "Investigate payment problems", "technical": "Fix product errors"},
        ),
        "urgent": Noul(instructions="The customer explicitly requests help today."),
        "detail": Score(
            instructions="How detailed is the request?",
            criteria=["Insufficient detail", "Problem identified", "Problem and evidence supplied"],
        ),
    })
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

`DecisionEngine.from_pretrained(model_path=None, *, device="auto", backend="auto",
strict=False, max_batch_tokens=8192, max_batch_size=32, cache_size=0,
offline=False, gpu_memory_utilization=0.2)` loads the pinned runtime model.
The default uses CPU ONNX Runtime without importing Torch or Transformers.

| Option | Meaning |
|---|---|
| `model_path` | Complete ONNX or native model directory; otherwise download the pinned files for the selected backend. With `backend="auto"`, a legacy directory containing `joint_config.json` but no `onnx_manifest.json` selects Torch. |
| `device` | `auto`, `cpu`, `cuda`, or `mps`. The default auto backend uses CPU; explicit `cuda`/`mps` selects Torch. Within the Torch backend, `auto` chooses CUDA, then MPS, then CPU. |
| `backend` | `auto` by default; chooses `onnx` for CPU, `torch` for explicit CUDA/MPS or a legacy native model directory. `onnx` is CPU-only; `torch` and `vllm` need their corresponding extras. |
| `strict` | Singleton scoring with ONNX/Torch; only native Torch preserves the reference execution shape. Unsupported with vLLM. |
| `max_batch_tokens` | ONNX/Torch padded-token budget per forward pass. |
| `max_batch_size` | ONNX/Torch inputs per forward pass; vLLM maximum concurrent sequences. |
| `cache_size` | Exact complete-input score-cache entries; 0 disables the cache. Range 0–1,000,000. |
| `offline` | When downloading implicitly, require an existing Hugging Face cache entry. Local-directory loading never needs a download. |
| `gpu_memory_utilization` | vLLM memory setting, strictly between 0 and 1; ignored by ONNX/Torch. |

| Method | Return |
|---|---|
| `engine.decide(state, candidates, *, question="Choose the best option.")` | `ChoiceAnswer`; `candidates` is a label-to-description mapping. |
| `engine.rank(state, candidates, *, question="")` | `RankingResponse`; accepts a mapping or list. |
| `engine.system_one(state, questions)` | `SystemOneResponse`; named `Choice`, `Noul`, or `Score` questions, or equivalent dictionaries. |
| `engine.metadata()` | Model revision, backend, device, limits, precision, and cache settings. |
| `engine.clear_cache()` | Clears the optional exact-input score cache. |

Reuse the engine across calls. Input validation, tokenization, and inference are
serialized within an engine; the HTTP server adds cross-request batching.
The state/question and each candidate are encoded jointly. A candidate-only
embedding cache would change this model's behavior.

## Question and answer types

Question constructors use keyword arguments. `instructions` is required, though
it may be any JSON-compatible value. String instructions and descriptions are
the clearest default.

| Question | Criteria | Answer |
|---|---|---|
| `Choice(instructions=..., criteria=...)` | Mapping of 2–64 labels to descriptions. | `ChoiceAnswer.choice` is the winning label. |
| `Noul(instructions=..., criteria=None)` | Optional descriptions under `false`/`true` or aliases `no`/`yes`. | `NoulAnswer.noul` is the normalized score weight for `true`, a float from 0 to 1. |
| `Score(instructions=..., criteria=...)` | Ordered list of 2–64 level descriptions. | `ScoreAnswer.score` is the expected zero-based level, `sum(i * p[i])`; it may be fractional. |

All answers contain `type`, `probabilities`, `confidence`, and raw `scores`.
`confidence` is the largest probability in the distribution. For `Noul` it is
**not necessarily** `noul`: when `false` is preferred, confidence is the false
probability. Score probability keys are string indices from `"0"` through
`str(len(criteria) - 1)`.

The response `answers` mapping preserves the names supplied in `questions`.
Both `SystemOneResponse` and `RankingResponse` also contain `model`, `usage`,
`temperature`, `probabilities_source`, and `latency_ms`.

`RankingResponse.ranked` contains objects with `rank` (starting at 1), `candidate`
(label), `text`, raw `score`, and normalized `probability`.

Scores are converted with softmax at the fixed temperature
`4.136820402388508`. They are relative ranking weights, not a guarantee of
correctness on new application data. See [limits and semantics](limits.md).

## HTTP clients

`DecisionClient(base_url=None, *, api_key=None, timeout=60, transport=None)` and
`AsyncDecisionClient(...)` expose `decide`, `rank`, `system_one`, and `health`.
Their inference method signatures match the engine methods, with additional
keyword arguments `timeout=None` and `extra_headers=None`.

```python
from gemmadecision import DecisionClient


def main():
    with DecisionClient("http://127.0.0.1:8700") as client:
        answer = client.decide(
            "I cannot sign in.",
            {"account": "Help with account access", "billing": "Help with payments"},
        )
        print(answer.choice)


if __name__ == "__main__":
    main()
```

Use `async with AsyncDecisionClient(...)` and await methods in async code.
Alternatively call `close()` / `await aclose()` explicitly. Clients reuse
connections, do not follow redirects, and do not retry decisions automatically.
HTTP error responses raise `httpx.HTTPStatusError`; transport failures raise
HTTPX request exceptions. Server routes and status codes are in the
[HTTP reference](http.md).

If omitted, `base_url` comes from `GEMMADECISION_BASE_URL`, falling back to
`http://127.0.0.1:8700`. The API key comes from `GEMMADECISION_API_KEY` unless
passed explicitly. `transport` is useful for HTTPX test transports.

## Native PydanticAI model

```bash
pip install 'gemmadecision[pydantic-ai]'
```

```python
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gemmadecision import GemmaDecisionModel


class Triage(BaseModel):
    team: Literal["billing", "technical"] = Field(description="Which team should handle the request?")
    urgent: bool = Field(description="Does the customer explicitly request attention today?")


def main():
    agent = Agent(GemmaDecisionModel.local(), output_type=Triage)
    result = agent.run_sync("I was charged twice. Please help today.")
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

| Constructor | Use |
|---|---|
| `GemmaDecisionModel.local(*, engine=None, settings=None)` | No HTTP server. Lazily shares the simple API's engine, or reuses an injected engine. |
| `GemmaDecisionModel(model_name="rajan2k/GemmaDecision-270M", *, base_url="http://127.0.0.1:8700", api_key=None, client=None, settings=None)` | Uses the HTTP service. An injected `AsyncDecisionClient` retains caller-owned connection lifetime. |

The native integration requires PydanticAI's decision-model support, installed
with the `pydantic-ai` extra. Supported output shapes and their expansion limits are
listed in [PydanticAI field constraints](limits.md#pydanticai-field-constraints).
Use field descriptions or agent `instructions` to express the question; the
prompt supplies the material being judged.

`result.response.provider_details` records decision metadata and a
`probabilities_source` of `softmax_scores`. PydanticAI's boolean confidence can
be a margin relative to its boolean threshold, so it is not interchangeable
with the wire answer's maximum probability. The model does not generate prose
or arbitrary JSON strings.

`settings` supports PydanticAI decision settings such as
`decision_boolean_threshold` (default 0.5). `timeout` is supported locally and
remotely. `extra_headers` is remote-only, and `extra_body` is rejected. A local
timeout does not forcibly interrupt a model operation already executing in a
worker thread. Close a remote model with `await model.aclose()` or `async with`;
an injected client/engine remains caller-owned.

## Download helper

`gemmadecision.model.download_model(*, local_dir=None, offline=False,
backend="onnx") -> Path` downloads the pinned public files with `token=False`
and returns their directory without instantiating an engine. Choose `torch`
or `vllm` to download the native weights instead.

ONNX loading checks the trusted export manifest and listed file hashes, including
its source v0.4.0 revision. Native loading checks the frozen inference files.
Copy the complete backend-specific directory for offline use.
