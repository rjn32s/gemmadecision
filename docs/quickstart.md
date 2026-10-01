# Quickstart

Make a local decision first. Add a server or framework when your application needs one.

## 1. Install

Use Python **3.11 or newer** in your application's environment:

```bash
python -m pip install gemmadecision
```

The standard install uses **ONNX Runtime on CPU**. It does not install Torch,
Transformers, the HTTP server, or PydanticAI. The pinned runtime model downloads
on first inference; later calls run locally with the loaded model.

For a notebook previously using 0.1.0, run `python -m pip install -U gemmadecision`
and restart the kernel. See the [notebook recovery guide](guides/troubleshooting.md#upgrade-a-010-notebook).

## 2. Choose an option

Save this as `example.py` and run `python example.py`:

```python
from gemmadecision import decide


def main():
    team = decide(
        "A payment appears twice on my statement.",
        choices={
            "billing": "Questions about payments, charges and refunds",
            "technical": "Problems with app crashes, connectivity or sign-in",
        },
        question="Which support team should handle this request?",
    )
    print(team)


if __name__ == "__main__":
    main()
```

The return value is one of the dictionary keys, as a string. A list also works:
`choices=["billing", "technical"]`. Descriptions let you define what each label means.

Keep the process alive when making several decisions: the default engine loads
once and is shared by `decide()`, `rank()` and local PydanticAI models. The first
call is slower. The [historical CPU measurements](cpu-latency.md) describe the
0.1.0 Torch runtime, not the current ONNX default.

## 3. Inspect the ranking

```python
from gemmadecision import rank


result = rank(
    "The app closes whenever I try to sign in.",
    choices={
        "billing": "Investigate payments and refunds",
        "technical": "Investigate app failures and sign-in errors",
        "account": "Change profile details and preferences",
    },
    question="Which queue best matches this request?",
)

for option in result.ranked:
    print(option.candidate, option.score, option.probability)
```

`ranked` is ordered from highest score to lowest. `candidate` is your label;
`text` is its description. With list input, `candidate` is a zero-based string
index and `text` contains the original option.

Probabilities are derived from ranking scores. They do not establish that an
answer is correct. [How scores work](concepts.md#scores-and-probabilities).

## 4. Ask several typed questions

Use an explicit engine when you want several question types in one call:

```python
from gemmadecision import Choice, DecisionEngine, Noul, Score


engine = DecisionEngine.from_pretrained()
result = engine.system_one(
    state="My receipt shows two charges for one order. Please help today.",
    questions={
        "team": Choice(
            instructions="Which team should receive this ticket?",
            criteria={
                "billing": "Payments, charges and refunds",
                "technical": "App failures and connection problems",
            },
        ),
        "time_sensitive": Noul(
            instructions="Does the customer explicitly ask for help today?",
        ),
        "priority": Score(
            instructions="Rate the requested response urgency.",
            criteria=["No deadline stated", "Help wanted soon", "Help explicitly wanted today"],
        ),
    },
)

print(result.answers["team"].choice)
print(result.answers["time_sensitive"].noul)  # Derived probability of yes, 0..1
print(result.answers["priority"].score)      # Expected rubric position, 0..2
```

Reuse `engine` for later calls. A score can be fractional; it is the expected
position across the ordered levels, not necessarily the most likely level.

## 5. Use PydanticAI

Install the integration when you need typed agents:

```bash
python -m pip install 'gemmadecision[pydantic-ai]'
```

```python
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gemmadecision import GemmaDecisionModel


class Ticket(BaseModel):
    team: Literal["billing", "technical"] = Field(description="Which support team handles this request?")
    time_sensitive: bool = Field(description="Does the customer explicitly ask for help today?")


agent = Agent(GemmaDecisionModel.local(), output_type=Ticket)
result = agent.run_sync("There are two payments for one order. Please help today.")
print(result.output)
```

Use `.local()` for in-process inference. `GemmaDecisionModel()` connects to a
separate HTTP server. See [PydanticAI](pydantic_ai.md) for async calls and supported schemas.

## 6. Start an API when you need one

In one terminal:

```bash
python -m pip install 'gemmadecision[serve]'
gemmadecision serve
```

In another terminal, make a request to the local server:

```bash
curl --fail-with-body http://127.0.0.1:8700/v1/rank \
  -H 'Content-Type: application/json' \
  -d '{"state":"The app closes when I sign in.","candidates":{"billing":"Payments and refunds","technical":"App failures and sign-in errors"},"question":"Which support queue fits?"}'
```

Open `http://127.0.0.1:8700/docs` for interactive API documentation.
[Serving, clients and authentication](guides/serving.md).

## Choose your next recipe

[Support routing](use-cases/ticket-routing.md) ·
[Evidence relations](use-cases/evidence-relation.md) ·
[Tool selection](use-cases/tool-selection.md) ·
[All use cases](use-cases/index.md)
