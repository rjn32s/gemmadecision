# Recipes

Give GemmaDecision the context, a question, and the possible answers. These
recipes show how to put that decision into an application using
`gemmadecision==0.1.0`.

```bash
pip install gemmadecision
```

Every example runs locally. There is no server to start or API key to create.
The first use downloads the published model; later runs reuse the download.
Keep an engine or agent alive to reuse its loaded weights across requests.
Each page has a complete Python example that you can save and run.

| I want to… | Start here | Interface |
|---|---|---|
| Send a support ticket to a team | [Ticket routing](ticket-routing.md) | `decide()` |
| Identify a customer's intent | [Intent classification](intent-classification.md) | `Choice` |
| Assign sentiment labels | [Sentiment](sentiment.md) | `Choice` |
| Ask a yes/no question | [Urgency check](urgency-check.md) | `Noul` |
| Assess a request against ordered levels | [Urgency rubric](urgency-rubric.md) | `Score` |
| Compare a claim with supplied evidence | [Evidence relation](evidence-relation.md) | `Choice` |
| Choose among answers I already have | [Response ranking](response-ranking.md) | `rank()` |
| Recommend a tool from an approved list | [Tool selection](tool-selection.md) | `Choice` |
| Return several typed decisions together | [PydanticAI triage](typed-triage.md) | `Agent` |
| Send uncertain decisions to a person | [Human review](human-review.md) | `Choice` + application rule |

These are illustrative workflows, not accuracy claims. The model's published
evaluations cover particular intent-routing and evidence-relation datasets;
they do not establish performance on every recipe or your application's data.
See the [model card](https://huggingface.co/rajan2k/GemmaDecision-270M) for the
evaluated tasks and limitations. Test representative examples from your own
domain before relying on a workflow.

The model ranks supplied alternatives. It does not write replies, extract
arbitrary text, or execute tools. Its probability fields come from normalized
ranking scores; they are not guaranteed probabilities of correctness. Adding
or removing alternatives can change those probabilities.

The examples use short inputs. Version 0.1.0 accepts 2–64 distinct candidates
per question, at most 32 questions and 256 candidate pairs per call, with
2,048 tokens for the state plus question and 768 tokens for each candidate.
Oversized inputs raise an error rather than being silently truncated.

For runnable examples in a repository checkout, explore the recipe launcher:

```bash
python examples/recipes.py --list
```

For an application calling a separate process or machine, keep the same
question objects and use the [HTTP client](../advanced.md).
