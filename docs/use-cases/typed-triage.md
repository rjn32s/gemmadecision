# Return typed triage with PydanticAI

Define the decisions as Pydantic fields. The native integration turns each
field into a question and returns an instance of your output model.

```python
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from gemmadecision import DecisionEngine, GemmaDecisionModel


class Triage(BaseModel):
    """Decide how a support ticket should enter the queue."""

    team: Literal["billing", "account", "technical", "review"] = Field(
        description=(
            "Which team should handle the ticket? Billing handles payments; "
            "account handles access; technical handles software problems; "
            "review handles unclear requests."
        )
    )
    urgent: bool = Field(
        description="Does the ticket give a deadline today or say work is currently blocked?"
    )


def build_triage_agent(engine: DecisionEngine):
    return Agent(GemmaDecisionModel.local(engine=engine), output_type=Triage)


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    agent = build_triage_agent(engine)
    result = agent.run_sync(
        "The application crashes when I upload a file, and I cannot finish today's report."
    )
    print(result.output.model_dump_json(indent=2))
    print(result.response.provider_details)


if __name__ == "__main__":
    main()
```

Reuse the engine and agent. The result has `.team` and `.urgent` fields with
the declared types. The questions are evaluated together, with one decision
per field; this is not a guarantee that all fields satisfy application-specific
relationships. Validate those relationships separately.

For the shortest setup, use `GemmaDecisionModel.local()` without passing an
engine. It loads the shared default model on first use. In an async application,
use `await agent.run(ticket)` instead of `run_sync()`.

Finite `Literal`, `Enum` and boolean fields fit this API. A free-form `str`
field such as `summary` is unsupported because this model does not generate
text. The [PydanticAI guide](../pydantic_ai.md) covers the available schemas.
