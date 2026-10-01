# gemmadecision

Small, local decisions in one Python call. [Available on PyPI](https://pypi.org/project/gemmadecision/). Powered by
[GemmaDecision-270M](https://huggingface.co/rajan2k/GemmaDecision-270M).

**[Documentation](https://rjn32s.github.io/gemmadecision/)** ·
[Quickstart](https://rjn32s.github.io/gemmadecision/quickstart/) ·
[10 use-case recipes](https://rjn32s.github.io/gemmadecision/use-cases/) ·
[API reference](https://rjn32s.github.io/gemmadecision/reference/python/)

## Install

```bash
pip install gemmadecision
```

Python 3.11+. No API key or server needed. Version 0.2.0 runs on CPU with
ONNX Runtime by default; installing it does not install Torch or Transformers.
The first call downloads the pinned runtime model; later calls reuse it.
After downloading, inference runs locally.

Upgrading from 0.1.0 in a notebook? Run `pip install -U gemmadecision`, then
restart the kernel. The new default avoids importing optional Torch vision/audio
packages. [Notebook upgrade guide](https://rjn32s.github.io/gemmadecision/guides/troubleshooting/#upgrade-a-010-notebook).

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

```bash
pip install 'gemmadecision[pydantic-ai]'
```

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
pip install 'gemmadecision[serve]'
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

- [0.2.0 lightweight runtime, conversion fidelity and CPU timings](https://rjn32s.github.io/gemmadecision/lightweight-runtime/)
- [Hardware settings, offline use, HTTP API and batching](https://github.com/rjn32s/gemmadecision/blob/main/docs/advanced.md)
- [vLLM backend for Linux/CUDA](https://github.com/rjn32s/gemmadecision/blob/main/docs/vllm.md)
- [CPU inference timings](https://github.com/rjn32s/gemmadecision/blob/main/docs/cpu-latency.md)
- [Performance measurements and reproduction](https://github.com/rjn32s/gemmadecision/blob/main/docs/performance.md)
- [Docker and releases](https://github.com/rjn32s/gemmadecision/blob/main/docs/releasing.md)

The historical 0.1.0 wheel passed fresh Modal CPU and H100 GPU checks for local
APIs, PydanticAI and HTTP serving. These checks used Torch and do not measure
the 0.2.0 ONNX default. [Installation validation](https://github.com/rjn32s/gemmadecision/blob/main/docs/installed-package-validation.md).

The default uses ONNX Runtime on CPU. Add only the integrations you need:

| Use | Install |
|---|---|
| Local CPU decisions | `pip install gemmadecision` |
| Rust HTTP server | `pip install 'gemmadecision[serve]'` |
| Native PydanticAI | `pip install 'gemmadecision[pydantic-ai]'` |
| Native Torch, CUDA or Apple MPS | `pip install 'gemmadecision[torch]'` |
| vLLM on supported Linux/CUDA | `pip install 'gemmadecision[vllm]'` |

Extras combine: `pip install 'gemmadecision[serve,torch]'` adds a GPU-capable
server. Select `--device cuda` or `--device mps` explicitly; in Python use
`DecisionEngine.from_pretrained(device="cuda")`. The default `auto` path uses
ONNX on CPU. No Rust compiler is needed for the HTTP server.

This model chooses among supplied options; it does not generate open-ended
text. Its scores are rankings, and the derived probabilities are not a guarantee
of correctness. Maximums are 2,048 state/question tokens, 768 tokens per choice
and 64 choices. See the [model card](https://huggingface.co/rajan2k/GemmaDecision-270M)
for evaluation and limitations.

Code: Apache-2.0. Model weights: separate Gemma terms. See [NOTICE](https://github.com/rjn32s/gemmadecision/blob/main/NOTICE).
