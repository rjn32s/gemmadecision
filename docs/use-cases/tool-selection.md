# Recommend a tool

Choose from a bounded list of capabilities the application has approved.
Describe what each tool is for and include a route for unclear requests.

```python
from gemmadecision import Choice, ChoiceAnswer, DecisionEngine


TOOLS = {
    "search_help": "Find product instructions or explanations in the help center.",
    "check_delivery": "Look up the delivery status of an existing order.",
    "human_review": "Ask a person because the request is unclear or needs a capability not listed.",
}


def recommend_tool(engine: DecisionEngine, request: str) -> ChoiceAnswer:
    response = engine.system_one(
        state=request,
        questions={
            "tool": Choice(
                instructions="Which listed capability is most relevant to this request?",
                criteria=TOOLS,
            )
        },
    )
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    return answer


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    answer = recommend_tool(engine, "My parcel has not arrived. Can you check its status?")
    print({"recommended_tool": answer.choice, "probabilities": answer.probabilities})


if __name__ == "__main__":
    main()
```

The result is a recommendation. This code does not call a tool, resolve its
arguments or authorize an action. Your application decides whether to proceed,
checks permissions and validates any arguments before execution. A selected
tool label should never be treated as authorization by itself.

To map a recommendation to a workflow, use an explicit allowlist of label-to-
handler mappings in your application; avoid evaluating generated code or
treating a returned label as a shell command.
