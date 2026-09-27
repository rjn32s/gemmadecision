# How decisions work

GemmaDecision ranks the alternatives you provide. Your application supplies the
context, asks a question and defines the available answers.

## State, question and choices

| Part | Purpose | Example |
| --- | --- | --- |
| State | Information available for this decision | A customer's message |
| Question | What the model should judge | Which support team handles this? |
| Choices | The permitted alternatives and their meaning | Billing: payments and refunds; technical: app failures |

State can be text or another JSON-compatible value. Keep it relevant to the
question. The model sees each candidate together with the state and question;
it does not retrieve missing facts or run tools to collect more information.

Use stable keys for application logic and descriptions to explain each option.
For example, `"billing": "Handle charges and refunds"` returns the key `billing`
when selected. The description helps define the task.

## Three question types

| Type | Input | Result |
| --- | --- | --- |
| `Choice` | A mapping of labels to descriptions | The highest-scoring label, with scores and derived probabilities |
| `Noul` | A yes/no question, optionally with criteria | A number between 0 and 1 representing the derived probability of yes |
| `Score` | Ordered descriptions of rubric levels | The expected zero-based rubric position |

For a three-level rubric, `Score.score` ranges from 0 to 2 and can be fractional.
It does not return the most likely level's label. If your application needs one
category, use `Choice` with three named levels instead.

Several questions can share a state in one `system_one()` call. Each produces
its own judgment. If one field constrains another, enforce that rule in your
application; these are not jointly constrained answers.

The simple `decide()` facade returns just a selected string. `rank()` gives you
the full ordered candidate list. The [Python reference](reference/python.md)
describes every result type.

## Scores and probabilities

Higher raw scores rank ahead of lower scores **within the same question and
candidate set**. Scores are not percentages. The package derives probabilities
using `softmax(scores / temperature)` with a fixed temperature of
`4.136820402388508`.

That temperature was fitted on 300 banking/NLI calibration examples. It does
not establish calibration for every application. Adding, removing or rewriting
choices changes the distribution. A reported probability of 0.9 is not evidence
of 90% correctness on your own data.

The direct engine and HTTP response's `confidence` is the largest derived
probability. PydanticAI may interpret distributions for its own decision types
and settings. See [PydanticAI](pydantic_ai.md) for boolean thresholds.

If you want automatic acceptance versus human review, measure outcomes on a
held-out set from your task and select a threshold from those results. You can
also provide an explicit `review` candidate. The [human-review recipe](use-cases/human-review.md)
shows that pattern without prescribing an untested cutoff.

## Local calls and serving

**Start locally:** `decide()`, `rank()` and `GemmaDecisionModel.local()` share a
lazy engine in the current Python process. The first inference downloads and
loads the pinned model; later calls reuse it. An explicit
`DecisionEngine.from_pretrained()` loads immediately and gives you hardware
and batching settings.

**Add HTTP when useful:** `gemmadecision serve` keeps one model process warm.
`DecisionClient`, `AsyncDecisionClient` and the remote `GemmaDecisionModel()`
connect to it. This is useful when several applications share a model or when
inference belongs on another machine.

Granian provides the Rust HTTP runtime. PyTorch or the optional vLLM backend
performs model computation. The [serving guide](guides/serving.md) covers the
queue, clients and authentication.

## What happens after a decision

The model returns a recommendation. Your code decides whether to route a ticket,
show a suggestion or execute an allowed action. It cannot write arbitrary prose
or extract unrestricted strings. A separate generative model can write a reply
after GemmaDecision selects a route.

The cookbook contains application examples, not accuracy guarantees for those
domains. Read [model scope and limits](reference/limits.md), then evaluate the
choices and inputs your application will actually use.
