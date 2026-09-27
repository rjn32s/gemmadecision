# gemmadecision

Small, local decisions in one Python call. Powered by
[GemmaDecision-270M](https://huggingface.co/rajan2k/GemmaDecision-270M).

## Install

```bash
pip install gemmadecision
```

Python 3.11+. No API key or server needed. The first call downloads the model
(about 0.5 GB); later calls reuse it. CPU, NVIDIA CUDA and Apple Silicon are
selected automatically. After downloading, inference runs locally.

## Use it in your code

```python
from gemmadecision import decide

team = decide("I was charged twice", choices=["billing", "technical"])
print(team)  # selected choice, as a string
```

For more specific decisions, give each choice a description:

```python
team = decide(
    "The same payment appears twice on my statement.",
    choices={
        "billing": "Handle charges, payments and refunds",
        "technical": "Handle crashes, login errors and app problems",
    },
    question="Which support team should handle this request?",
)
```

Need the full ranking? Use `rank(...)` with the same arguments. It returns
ordered candidates, scores and derived probabilities.

## PydanticAI

```python
from typing import Literal
from pydantic_ai import Agent
from gemmadecision import GemmaDecisionModel

agent = Agent(
    GemmaDecisionModel.local(),
    output_type=Literal["billing", "technical"],
)
result = agent.run_sync("I was charged twice")
print(result.output)
```

This uses PydanticAI's native decision-model interface. Boolean, enum, rubric
and finite Pydantic fields are supported.
[More PydanticAI examples](https://github.com/rjn32s/gemmadecision/blob/main/docs/pydantic_ai.md).

## Serve it

```bash
gemmadecision serve
```

That starts the Rust-powered Granian HTTP server at `http://127.0.0.1:8700`.
Interactive API documentation is available at `/docs`.

```python
from gemmadecision import DecisionClient

with DecisionClient() as client:
    result = client.decide(
        "I was charged twice",
        candidates={"billing": "Payments and refunds", "technical": "App problems"},
    )
    print(result.choice)
```

`AsyncDecisionClient` works with `await`. To connect PydanticAI to the server,
use `GemmaDecisionModel()` instead of `.local()`.

## More control when you need it

- [Hardware settings, offline use, HTTP API and batching](https://github.com/rjn32s/gemmadecision/blob/main/docs/advanced.md)
- [vLLM backend for Linux/CUDA](https://github.com/rjn32s/gemmadecision/blob/main/docs/vllm.md)
- [Performance measurements and reproduction](https://github.com/rjn32s/gemmadecision/blob/main/docs/performance.md)
- [Docker and releases](https://github.com/rjn32s/gemmadecision/blob/main/docs/releasing.md)

The default uses batched PyTorch for model computation and Rust for HTTP
serving. vLLM is optional; no Rust compiler is needed to install the package.

This model chooses among supplied options; it does not generate open-ended
text. Its scores are rankings, and the derived probabilities are not a guarantee
of correctness. Maximums are 2,048 state/question tokens, 768 tokens per choice
and 64 choices. See the [model card](https://huggingface.co/rajan2k/GemmaDecision-270M)
for evaluation and limitations.

Code: Apache-2.0. Model weights: separate Gemma terms. See [NOTICE](https://github.com/rjn32s/gemmadecision/blob/main/NOTICE).
