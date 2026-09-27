# Troubleshooting

Start with the installed version and runtime information:

```bash
gemmadecision --version
gemmadecision doctor
```

`doctor` reports installed libraries without loading the model. Server logs
appear in the terminal running `gemmadecision serve`.

## The first call is taking longer

The first local call or server start downloads roughly 0.5 GB of model files
and loads the inference runtime. GPU setup may also initialize kernels. Warm
inference measurements exclude that work.

Download explicitly to separate download progress from startup:

```bash
gemmadecision download --output ./model
gemmadecision serve --model-path ./model --offline
```

For an HTTP server, wait until `GET /ready` succeeds. For local Python use,
keep the process running and reuse the default engine, or construct one
`DecisionEngine` and reuse it. Starting a new Python process for every decision
reloads the model each time.

If model verification fails, keep the complete pinned download intact. Point
the server at a fresh `gemmadecision download --output` directory instead of
mixing a head, tokenizer or encoder from another model release.

## HTTP errors

| Status | What happened | What to check |
|---|---|---|
| `401` | The inference key is missing or incorrect | Match the server's key in the client or `Authorization: Bearer ...` header |
| `413` | The POST body exceeds 2 MiB | Send a smaller state and fewer questions |
| `422` | Invalid schema, duplicate/empty candidates or an input limit | Read the response's `detail`; shorten the input or fix the request |
| `503` | The inference queue is full, or the readiness endpoint reports an unavailable worker | Reduce caller concurrency; inspect server logs and `/ready` |
| `504` | The server's decision deadline elapsed | Reduce the workload or raise `--request-timeout` after checking latency |
| `500` | Inference failed | Inspect the server-side exception and available device memory |

The Python clients raise `httpx.HTTPStatusError` for HTTP failures. A client
timeout can occur before a server timeout if the client's deadline is shorter.
For example, increase the client deadline with `DecisionClient(timeout=120)`
only when the workload needs it. The clients do not automatically retry an
inference request; a timed-out operation that has already started may still
finish on the server.

### Input limits

The serving contract accepts:

- 2,048 tokens for the rendered state and question together.
- 768 tokens per candidate description.
- 2–64 distinct candidates per question.
- At most 32 questions and 256 total candidate pairs per request.
- A POST body of at most 2 MiB.

Token counts include tokenizer special tokens. The package refuses overlong
inputs instead of silently truncating them. A custom, smaller
`--max-batch-tokens` setting can also reject an otherwise valid joint input.
Increasing a queue size does not extend input limits.

### Authentication succeeds on health but fails on inference

Health, readiness, model information and API documentation are readable without
an inference key. Protected calls to `/v1/systemone` and `/v1/rank` still need
the configured key. SDK clients read `GEMMADECISION_API_KEY`; curl needs the
bearer header. See [network serving](serving.md#connect-from-another-machine).

## The command uses the wrong Python environment

Install and run through the same interpreter:

```bash
python -m pip install gemmadecision
python -m gemmadecision.cli doctor
python -m gemmadecision.cli serve --device cpu
```

Check `python -m pip show gemmadecision` if a different shell or notebook cannot
import the package. In a notebook, install into the interpreter used by its
kernel. The package supports Python 3.11 and later; the selected tensor backend
must also provide compatible wheels for that Python version and platform.

For CUDA errors, first confirm that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Use `--device cpu` to select CPU execution explicitly. The CPU Dockerfile
installs CPU PyTorch, while the optional vLLM backend requires its own supported
Linux/CUDA environment. See [deployment](deployment.md).

## PydanticAI rejects my output type

The model selects among finite options. Start with a supported output type:

```python
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gemmadecision import GemmaDecisionModel

class TicketRoute(BaseModel):
    team: Literal["billing", "technical", "review"] = Field(
        description="Which team should handle this ticket?"
    )
    urgent: bool = Field(description="Does the ticket need prompt attention?")

agent = Agent(GemmaDecisionModel.local(), output_type=TicketRoute)
result = agent.run_sync("The same payment appears twice.")
print(result.output)
```

Unrestricted `str` fields, arbitrary dictionaries and unbounded numeric outputs
cannot be filled by choosing among known candidates. Use finite Literal/Enum
options, booleans or a described rubric, and handle generated prose or free-text
extraction in a separate component. The [PydanticAI guide](../pydantic_ai.md)
explains the supported schemas.

`GemmaDecisionModel.local()` loads the model in your Python process.
`GemmaDecisionModel()` connects to an HTTP server. If you see a connection error
while expecting local inference, check which constructor you used.

## The answer is a valid label, but the wrong one

Schema validation establishes the output type, not decision accuracy. Make
option descriptions concrete and distinct, include an explicit review option
when appropriate, and evaluate on representative labeled examples. See
[choosing options](choosing-options.md) for probabilities, thresholds and
none-of-the-above behavior.
