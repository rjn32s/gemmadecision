# Choose useful options

GemmaDecision compares the candidates you supply. A good request has a clear
question, enough relevant context and descriptions that distinguish the
available actions.

## Keep labels stable and descriptions specific

Labels belong to your application. Descriptions tell the model what each label
means:

```python
from gemmadecision import decide

team = decide(
    "The same card payment appears twice on my statement.",
    choices={
        "billing": "Investigate charges, duplicate payments and refund requests.",
        "technical": "Investigate app crashes, login failures and software errors.",
        "review": "Clarify requests that do not fit either team or lack enough information.",
    },
    question="Which support team should handle this request?",
)
print(team)
```

The returned value is a label such as `billing`. The description is what enters
the model's joint state/question/candidate input. A label such as `queue_17`
can work if its description explains the action; a label alone does not tell
the model your internal business rules.

For a list of strings, the simple `decide()` helper uses each string as both
label and description. For `rank()` with a list, returned labels are string
indices such as `"0"` and `"1"`. Use a mapping when your application needs
explicit identifiers.

## Describe real alternatives

Give every option a distinct, nonempty description. Avoid offering two labels
with the same description, such as `"a": "Help the customer"` and
`"b": "Help the customer"`; the package rejects duplicate descriptions.

Keep options at a comparable level of detail. If one option is a team and
another is a specific multi-step procedure, make the question explain which
kind of decision you need. Put shared facts in the state rather than repeating
them in every description. Put the decision criterion in the question.

Descriptions can explain constraints, but the output remains a model judgment.
Enforce hard rules such as access permissions or mandatory human review in
application code.

## Include a review or none-of-the-above option

A choice request always ranks the supplied options. If your application needs
to abstain, include an actual candidate that describes that outcome:

```python
choices = {
    "refund": "The information supports investigating a payment refund.",
    "login": "The information supports investigating account access.",
    "needs_review": "Neither action fits, or the available information is insufficient.",
}
```

This makes review selectable; it does not guarantee that every uncertain input
will be routed there. Test review behavior alongside the other choices.

For a `Choice` question, a `None` description in Python, or `null` in JSON,
means **use the label itself as its description**. It does not mean “reject all
options” and does not produce a null answer. Use an explicit description for
your review label. The rank endpoint requires string descriptions.

## Inspect rankings when you need them

```python
from gemmadecision import rank

result = rank(
    "The same card payment appears twice.",
    choices={
        "billing": "Investigate duplicate charges and refunds.",
        "technical": "Investigate app crashes and login errors.",
    },
    question="Which support team should handle this?",
)
for item in result.ranked:
    print(item.candidate, item.score, item.probability)
```

Higher raw scores rank ahead of lower ones. Scores are not percentages and
should not be compared as confidence across unrelated requests. The public
API transforms each candidate set with a fixed softmax temperature to produce
probabilities. Adding or removing alternatives changes that distribution.

`confidence` is the highest derived probability. It is not a measured
probability that the answer is correct on your data. Choose any acceptance or
review threshold using held-out examples from your own application. Changing
a threshold changes routing behavior; it does not recalibrate or retrain the
model.

## Rubric scores are expectations

Use `Score` when you have ordered, described levels:

```python
from gemmadecision import DecisionClient, Score

with DecisionClient() as client:
    result = client.system_one(
        state="The customer cannot sign in and cannot complete today's work.",
        questions={
            "priority": Score(
                instructions="How much attention does this request need?",
                criteria=[
                    "Routine question; work can continue.",
                    "A problem needs attention, but a workaround is available.",
                    "The customer is blocked from completing their work.",
                ],
            )
        },
    )
    answer = result.answers["priority"]
    print(answer.score)
    most_likely_level = max(answer.probabilities, key=answer.probabilities.get)
    print(most_likely_level)
```

Start the [HTTP server](serving.md) before running this client example.
Levels are numbered from zero. `score` is the probability-weighted expected
index, so three levels can yield a fractional result such as `1.4`. The most
likely level is the key with the highest probability, which is a different
quantity. Numeric text inside a level description does not change the index
used in the expectation.

For practical limits and invalid-input behavior, see
[troubleshooting](troubleshooting.md). For finite typed outputs directly in
your application, see [PydanticAI](../pydantic_ai.md).
