<span class="hero-kicker">gemmadecision · Python package 0.2.0</span>

# Small decisions. A few lines of Python.

<p class="hero-lead">Choose a support queue, classify a request, score an ordered rubric, or return a typed decision. Run GemmaDecision-270M locally, directly from your application.</p>

[Start in Python](quickstart.md){ .md-button .md-button--primary }
[Browse use cases](use-cases/index.md){ .md-button }

```bash
pip install gemmadecision
```

```python
from gemmadecision import decide

team = decide(
    "A payment appears twice on my statement.",
    choices=["billing", "technical"],
)
print(team)
```

The first call downloads and loads the public model. Later calls reuse it in the
same process. The default uses ONNX Runtime on CPU without installing Torch
or Transformers. No hosted-model API key or server is required for local use.

<div class="grid cards" markdown>

- **Use it in an application**

    Return a label with `decide()`, or inspect all candidates with `rank()`.

    [Python quickstart](quickstart.md)

- **Keep outputs typed**

    Install the `pydantic-ai` extra for `Literal`, `Enum`, boolean and finite Pydantic fields.

    [PydanticAI guide](pydantic_ai.md)

- **Share one model over HTTP**

    Install the `serve` extra, then start `gemmadecision serve` and connect clients.

    [Serving guide](guides/serving.md)

- **Choose a deployment**

    Start on CPU with ONNX Runtime. Add the Torch extra for CUDA or Apple MPS; vLLM is optional on Linux/CUDA.

    [Deployment guide](guides/deployment.md)

</div>

## Pick the output you need

| Your application needs | Start here |
|---|---|
| One label from a known set | [Route a support ticket](use-cases/ticket-routing.md) |
| A yes/no signal | [Check urgency](use-cases/urgency-check.md) |
| A score on described levels | [Score urgency](use-cases/urgency-rubric.md) |
| Several fields in one response | [Typed triage](use-cases/typed-triage.md) |
| Ordered supplied options | [Rank responses](use-cases/response-ranking.md) |
| A review path for uncertain decisions | [Human review](use-cases/human-review.md) |

## What has been verified

The historical **0.1.0 Torch release** passed fresh Modal CPU and H100 checks
for direct Python calls, local PydanticAI and real HTTP serving. Those checks
do not establish 0.2.0 ONNX performance. On four CPU cores, two
short 30-token candidate inputs took **124 ms median** in a small repeated
measurement. [CPU timings](cpu-latency.md) and
[installation evidence](installed-package-validation.md) include the scope and
raw results.

The recipes demonstrate ways to integrate a closed-choice ranker. Accuracy in
each new application needs its own evaluation; the released model's measured
domains are banking-support topics and three-way evidence relations. It does
not generate free-form text. [Model scope and limits](reference/limits.md).

[PyPI package](https://pypi.org/project/gemmadecision/) ·
[Model card and weights](https://huggingface.co/rajan2k/GemmaDecision-270M) ·
[GitHub source](https://github.com/rjn32s/gemmadecision)
