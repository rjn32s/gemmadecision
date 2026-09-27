# PydanticAI: typed decisions from a local model

Start with in-process inference. Install the model runtime and integration:

```bash
pip install gemmadecision
```

There is no server to start, and no external model account or paid API key:

```python
from typing import Literal
from pydantic_ai import Agent
from gemmadecision import GemmaDecisionModel

router = Agent(
    GemmaDecisionModel.local(),
    output_type=Literal['billing', 'account', 'technical'],
    instructions='Which support team should handle this ticket?',
)

result = router.run_sync('I was charged twice.')
print(result.output)
```

The constructor does not download or load weights. The first request downloads
the pinned model once and loads it; subsequent calls reuse it. This is the same
process-local engine used by `gemmadecision.decide()` and `gemmadecision.rank()`.
The local model also supports `await router.run(...)`: loading and inference run
in a worker thread, keeping the event loop responsive. Engine locks serialize
tokenizer and model access. This path does not implement server microbatching.

For explicit hardware settings, create `DecisionEngine.from_pretrained(...)`
and pass it to `GemmaDecisionModel.local(engine=engine)`. The process or caller
owns the engine: exiting the model's async context does not unload shared
weights. A timeout stops waiting and skips inference if it has not started;
an already running model operation may finish in the background.

## Use an existing server

For a separate server, use the remote constructor. Start the server with the main
README's serving command:

```python
import asyncio
from typing import Literal

from pydantic_ai import Agent
from gemmadecision.integrations.pydantic_ai import GemmaDecisionModel

async def main():
    async with GemmaDecisionModel() as model:
        router = Agent(
            model,
            output_type=Literal['billing', 'account', 'technical'],
            instructions='Which support team should handle this ticket?',
        )
        result = await router.run('I was charged twice.')
        print(result.output)
        print(result.response.provider_details['probabilities'])

asyncio.run(main())
```

The output is one of your supplied labels. The integration subclasses
PydanticAI's public `DecisionModel` interface, which converts finite output
schemas into ranking questions. It does not wrap a ranker as a text generator.

## Several decisions in one request

Each field becomes a question; questions are evaluated together in one local
engine call, or one HTTP request with a remote server. Use field descriptions
for precise questions and enum member
descriptions when the option names alone are ambiguous.

```python
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    team: Literal['billing', 'account', 'technical'] = Field(
        description='Which team should handle this ticket?'
    )
    urgent: bool = Field(description='Does the customer need attention today?')

agent = Agent(GemmaDecisionModel.local(), output_type=Ticket)
result = agent.run_sync('My card was charged twice. I need a refund today.')
print(result.output.team)
print(result.output.urgent)
```

Supported PydanticAI decision schemas include bools, finite Literal/Enum
choices, nested models of decision fields, and described rubric levels.
Free-form `str` fields, arbitrary dictionaries, and unbounded numeric outputs
are unsupported. Each choice or rubric allows at most 64 options, with at most
32 questions and 256 candidate pairs per request; input limits still apply. A list of labels becomes
one yes/no question per label and counts toward that question limit.
PydanticAI validates unsupported output schemas before sending a
request. Multiple questions are independent judgments, not joint constrained
generation, so enforce cross-field business rules in your application.

## Reuse clients and route work

Set `GemmaDecisionModel(base_url='http://your-server:8700', api_key='...')` for
another server. Or pass `client=existing_async_client` to reuse its pool. An
injected client belongs to its caller and is not closed by the model. Requests
honor PydanticAI's `timeout` and `extra_headers` settings. `extra_body` is
rejected to keep the decision protocol explicit. Local models accept a timeout
but reject `extra_headers`, which has no meaning without an HTTP request.

PydanticAI also supports typed output functions, tools and `FallbackModel`
composition with this native interface. A decision selects a finite route or
fills its finite typed fields; a separate generative model can handle a route
that requires writing text. If tools perform real actions, the application
remains responsible for its normal authorization rules.

```python
def route(team: Literal['billing', 'account', 'technical']) -> str:
    """Select the queue for this ticket.

    Args:
        team: Which team should handle this ticket?
    """
    return f'Put this ticket in the {team} queue'

agent = Agent(GemmaDecisionModel.local(), output_type=route)
result = agent.run_sync('I was charged twice.')
```

## Confidence and performance

`provider_details` includes per-field confidence/distributions and
`probabilities_source='softmax_scores'`. These numbers are probabilities derived
from ranking scores. They are **not measured probabilities of correctness on
your data**. Choose thresholds using held-out examples from your application.
Changing `decision_boolean_threshold` changes the yes/no decision cutoff; it
does not retrain the model or turn its scores into calibrated confidence.

The model produces the decision in one response. PydanticAI's streaming API
works by emitting the complete answer; it does not make inference faster or
provide earlier partial tokens. Reuse a warm local model or server for low latency.
The Python client and PydanticAI perform orchestration; inference speed comes
from the serving backend and hardware.

## Compatibility and testing

The package includes `pydantic-ai-slim>=2.51,<3` and was tested against
2.51.0. PydanticAI is imported only when the integration module is used.
Tests exercise actual PydanticAI agents against deterministic fake HTTP
clients and local inference backends, including lazy shared loading, typed
results, one-call multi-field decisions, streaming, output-function routing,
response validation, timeouts, errors and client ownership.
Those tests verify integration behavior, not model accuracy.

Reference: [PydanticAI decision model documentation](https://pydantic.dev/docs/ai/models/decision/).
