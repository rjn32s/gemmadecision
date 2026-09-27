# Limits and result semantics

GemmaDecision chooses among alternatives you provide. It does not generate
prose, extract arbitrary strings, or decide what tools your application is
authorized to execute. This page describes the published 0.1.0 package contract.

## Input limits

| Item | Limit |
|---|---:|
| Candidate choices in one ranking or choice question | 2–64 |
| Levels in one score question | 2–64 |
| Named questions in one request | 1–32 |
| Candidate pairs across all questions in one request | 256 |
| Rendered state plus question instructions | 2,048 tokens |
| Each candidate description | 768 tokens |
| HTTP POST body | 2 MiB / 2,097,152 bytes |
| Default Torch batch size | 32 joint inputs |
| Default Torch batch budget | 8,192 padded tokens |
| Default server queue capacity | 128 waiting requests |
| Default server request deadline | 60 seconds |

Token limits include tokenizer special tokens. They apply to rendered input,
so nested state and structured instructions count toward the same 2,048-token
state/question budget. Every candidate is concatenated with that state/question
and a fixed separator, then checked against the backend's complete-input limit.
The Torch encoder's larger architectural context does not increase the
published state or candidate limits. vLLM uses a 3,072-token complete-input
limit by default.

Text is never silently truncated. Invalid local requests raise validation
errors; HTTP requests return 422. Candidate descriptions must be distinct and
nonempty after rendering. For example, a request with five 64-option questions
exceeds the 256-pair limit even though each question is individually valid.

A single Torch joint input must fit the configured `max_batch_tokens`, since
even a one-element batch must respect that budget. The scheduler can split a
larger group of valid inputs into multiple forwards. It does not treat the
batch-token setting as a larger context window.

## Scores, probabilities, and confidence

The model produces one raw scalar per joint input. Larger scores rank ahead
of smaller scores. They are uncalibrated ranking scores, and values from
different requests should not be compared as a common confidence scale.

The API applies softmax across a question's candidates at a fixed temperature
of `4.136820402388508`. Consequently:

- Probabilities sum to one within that candidate set.
- Adding or removing candidates changes the normalized probabilities.
- A large probability is not a universal estimate that the decision is correct.
- An application fallback cutoff should be selected on its own held-out data.

The wire `confidence` field is the maximum probability. `Noul.noul` is the true
outcome's probability, while `Score.score` is the expected zero-based rubric
level. A score of 1.4 on a three-level rubric is valid; it is not necessarily
the index of the most probable level. Exact raw-score ties preserve input order.

Batch shape and numeric kernels can introduce small floating-point differences.
Torch `strict=True` preserves the published singleton batch shape for
comparisons, but it does not promise bit-identical results across devices or
software versions. See [measured numerical agreement](../performance.md).

## PydanticAI field constraints

`GemmaDecisionModel` uses PydanticAI's decision-model translation. These field
shapes are supported inside a Pydantic output model by the tested PydanticAI
2.51 integration. Simple types such as `bool` and `Literal` can also be used
directly as the agent's `output_type`:

| Output shape | Translation / constraint |
|---|---|
| `bool` | One yes/no question; default boolean threshold is 0.5. |
| `Literal["a", "b", ...]` or string `Enum` | A choice with at least 2 and at most 64 alternatives. |
| Whole-number `Literal` / `Enum` | Finite categorical choices; arbitrary unbounded `int` is not supported. |
| Described levels `0, 1, …, N-1` | A rubric only when every numeric level has a description in the JSON schema. Otherwise they are categorical choices. Maximum 64 levels. |
| `Annotated[float, Field(ge=0, le=1)]` | A yes/no probability returned as a float, without boolean thresholding. A positive upper bound can also scale the units, e.g. 100 for a percentage; stepped `multiple_of` constraints are unsupported. |
| `list[Literal["a", "b", ...]]` or list of string `Enum` | One yes/no question per option, returning selected options. At least 2 string options; no list-length constraints. |
| `dict[Literal["a", "b", ...], bool]` | One yes/no question per finite key, retaining every key. Arbitrary free-form dictionary keys/values are unsupported. |
| Pydantic model of supported fields | Fields become named questions; nested models expand to their supported leaf fields. |
| Optional finite categorical field | Adds a distinct “none of these” choice. Optional booleans, probability fields, and rubrics are not interchangeable with this form. |

A rubric returned through the native typed model is rounded to a valid level
(half values round upward). The wire `ScoreAnswer.score` retains the fractional
expectation. Native decision metadata records this difference.
Put a described numeric rubric in a Pydantic model field: a bare union passed
as `output_type` can instead be interpreted as separate output routes.

Expanded questions still share the package's **32-question and 256-pair**
limits. A list with many alternatives can consume multiple question slots;
the limit is not simply the number of Pydantic fields.

Free-form `str`, unconstrained numeric output, `list[str]`, arbitrary objects,
and generative explanations are not expressible as ordinary decision fields.
Use a `Literal`, `Enum`, or another supported decision shape instead. Give
fields descriptions and use agent `instructions` to state what is being asked;
the user prompt carries the material to classify.

PydanticAI can also route among finite output types or tools. A tool whose
arguments cannot be represented may cause a handoff; it does not turn this
model into a text generator. The optional `decision_route_threshold` applies
to route selection, not as a universal confidence check for a single typed
answer. Keep tool authorization and side-effect policy in the application.

## Runtime and memory

Python 3.11 or newer is required. The normal install includes Torch serving and
PydanticAI. vLLM is optional and pinned to 0.30.0 for the supported pooling path.

Torch supports CPU, CUDA, and MPS. Its encoder uses BF16 on CUDA when supported,
and FP32 otherwise; normalization and the scalar head are FP32. vLLM requires a
BF16-capable CUDA GPU; its encoder is BF16 and its scalar head runs FP32 on CPU.
The published CPU and GPU measurements are hardware-specific, not a memory or
latency guarantee for another machine.

The server starts one model process. Its Rust HTTP layer does not replace the
Torch/vLLM tensor computation. Cache entries store scores for exact complete
joint inputs, never independent state and candidate embeddings. The cache is
off by default.

## Usage and cancellation

`usage.input_tokens` counts actual encoded joint inputs, including repeated
state/question tokens for distinct candidates. Inputs reused through the score
cache or deduplicated within a grouped batch are not charged again in that
count. `cached_pairs` includes those reused pairs. Token accounting may therefore
be assigned to the first request that encodes a shared input in a batch.

`output_tokens` is zero because ranking does not generate tokens. `total_tokens`
equals `input_tokens`. These are serving-work counters, not a model-provider
billing tariff.

Neither the SDK nor the server automatically retries inference. An HTTP or
local native-model timeout stops waiting for the answer; an already running
tensor operation may still finish. Check the HTTP [status reference](http.md)
and configure application retry/fallback behavior explicitly.
