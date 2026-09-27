# gemmadecision

Typed local decisions for Python and PydanticAI. One command serves the complete
[GemmaDecision-270M](https://huggingface.co/rajan2k/GemmaDecision-270M) model with
Granian's **Rust HTTP runtime**, bounded request queues and batched inference.
Importing the client does not load PyTorch or download weights.

```sh
pip install gemmadecision
gemmadecision serve
```

The first server start downloads the pinned public model (about 0.5 GB). Subsequent
starts use the cache. CPU, CUDA and Apple MPS are selected automatically; override
with `--device cpu`, `--device cuda`, or `--device mps`. Model inference stays local.
The client and native PydanticAI integration work without an external LLM API key.

## Python

```python
from gemmadecision import DecisionClient

with DecisionClient() as client:
    answer = client.decide(
        "The same card payment appears twice on my statement.",
        candidates={
            "billing": "Investigate duplicate payments, charges and refunds.",
            "technical": "Investigate app crashes and login errors.",
        },
        question="Which support team should handle this request?",
    )
    print(answer.choice)
    print(answer.probabilities)
```

Use `AsyncDecisionClient` with `async with` and `await` in async applications.
Clients reuse HTTP connections and do not retry decisions automatically. Set
`base_url`, `api_key` and `timeout` explicitly or use `GEMMADECISION_BASE_URL` and
`GEMMADECISION_API_KEY`. The standard install includes the local runtime; importing only the client stays lightweight.

## PydanticAI, directly

PydanticAI has a native decision-model interface: no chatbot shim is needed.

```python
import asyncio
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gemmadecision.integrations.pydantic_ai import GemmaDecisionModel

class Route(BaseModel):
    department: Literal["billing", "technical"] = Field(
        description="Team responsible for this request: billing handles payments; technical handles app failures."
    )

async def main():
    async with GemmaDecisionModel() as model:
        agent = Agent(model, output_type=Route)
        result = await agent.run("The same card payment appears twice on my statement.")
        print(result.output)

asyncio.run(main())
```

Finite `Literal`, `Enum`, boolean and rubric outputs are supported. This model
selects among supplied alternatives; it does not generate arbitrary prose,
extract unrestricted strings or replace a general language model. See
[PydanticAI integration](pydantic_ai.md) for routing, rubrics and fallbacks.

## Several typed decisions in one request

```python
from gemmadecision import DecisionClient, Choice, Noul, Score

with DecisionClient() as client:
    result = client.system_one(
        state="The app crashed three times today and I cannot sign in.",
        questions={
            "team": Choice(instructions="Who should handle this?", criteria={
                "technical": "App crashes and login problems", "billing": "Charges and refunds"}),
            "needs_help": Noul(instructions="Does this customer need assistance?"),
            "urgency": Score(instructions="How urgent is the request?",
                             criteria=["Can wait", "Needs attention", "Blocked and needs prompt help"]),
        },
    )
    print(result.answers)
```

`client.rank(state, candidates, question=...)` exposes raw scores and their fixed
softmax transformation. For in-process inference use
`DecisionEngine.from_pretrained(device="auto")` with the same `decide`, `rank` and
`system_one` methods. It loads only when explicitly constructed.

## Serving choices

- **Default:** Granian Rust HTTP + batched PyTorch. Length-bucketed candidate
  batches, bounded cross-request microbatching, one model process, no generation.
- **Reference:** `gemmadecision serve --strict --batch-wait-ms 0` scores each
  candidate separately, matching the published numerical execution shape.
- **vLLM:** `pip install 'gemmadecision[vllm]'` in a separate Linux CUDA environment,
  then `gemmadecision serve --backend vllm --device cuda`. See the exact dependency
  pins and evaluation status in [vLLM setup](vllm.md).
- **Offline:** `gemmadecision download --output ./model`, then
  `gemmadecision serve --model-path ./model --offline`.
- **Docker:** see [release and container instructions](releasing.md).

The serving process uses Rust for HTTP transport and Rust-backed Pydantic
validation. Model tensor operations use PyTorch or vLLM's native CUDA kernels;
this is not an all-Rust reimplementation of Gemma. Granian supplies maintained
binary wheels so users do not need to compile a new bespoke Rust server.

CLM's vLLM encoder/API design inspired this separation. This model is a **joint
ranker**: the entire state, question and candidate must be encoded together.
Caching independent action embeddings would change its predictions. Optional
`--cache-size 1024` caches only exact full-input scores in memory; it is off by
default. Cache keys are SHA256 digests, and no persistent prompt cache is written.

`--max-batch-tokens`, `--max-batch-size`, `--batch-wait-ms` and `--queue-size`
control memory and throughput. Start with the defaults. Batching can introduce
small numerical differences, especially near ties; see [performance notes](performance.md).
No universal speed multiplier or JEV-equivalent accuracy is claimed.

## HTTP interface

`POST /v1/systemone`, `POST /v1/rank`, `GET /v1/models`, `GET /health`, `GET /ready`.
Interactive OpenAPI documentation is at `http://127.0.0.1:8700/docs`.
The default bind address is loopback. To expose on a private network, set
`GEMMADECISION_API_KEY` and use `--host 0.0.0.0`. Put a TLS reverse proxy in front
of any internet-facing deployment. An explicit `--allow-unauthenticated` flag
exists for controlled deployments. HTTP request-content logs are disabled.

Maximum input: 2,048 state/question tokens; 768 tokens per candidate; 2–64
distinct candidates per question; 32 questions and 256 total pairs per request;
2 MiB request body. Inputs are refused with HTTP 422 rather than silently
truncated. Overload returns 503; queue timeout returns 504. Health is independent
of inference authorization and includes no request text. There is one bounded
inference queue and one model instance per process.

## Model identity and limitations

Package version `0.1.0` is separate from model version `v0.4.0`. Every backend
uses immutable model revision `785d530221c990671f29976902540101bb9c7647` and verifies
the encoder/head/config hashes. No executable Hugging Face code is downloaded
or enabled. The encoder was trained jointly with a scalar head; using stock
Gemma token-generation logits would not serve this model.

Probabilities are `softmax(raw_scores / 4.136820402388508)`, using a temperature
previously fitted on 300 banking/NLI examples. `confidence` is the largest
derived probability, not a guarantee of correctness or general calibration.
On the author's public JevBench diagnostic the frozen model achieved 110/231
(47.62%), including 36 context-limit refusals. It is most relevant to small
closed-choice tasks; evaluate it on your own use case before routing important
actions. [Model evaluation and provenance](https://huggingface.co/rajan2k/GemmaDecision-270M).

## Develop

```sh
git clone https://github.com/rjn32s/gemmadecision.git
cd gemmadecision
pip install -e '.[test,pydantic-ai]'
pytest
python -m build
twine check dist/*
```

Tests use synthetic tensors, stub encoders and actual PydanticAI agents with
mock HTTP; they do not download or train a model. GPU performance measurements
are separate reproducible artifacts. `gemmadecision doctor` reports installed
backends without loading weights.

Code: Apache-2.0. Weights: separate Gemma terms. Dependency licenses and CLM schema
attribution are in [NOTICE](NOTICE). This package is not affiliated with Google,
PydanticAI, vLLM, Granian, CLM or the JevBench maintainers.
