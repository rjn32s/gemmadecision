"""Typed local classification; requires a running GemmaDecision server.

    pip install 'gemmadecision[pydantic-ai]'
    python examples/pydantic_ai_router.py

No generative-model API key is needed. See the README for starting the server.
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from gemmadecision.integrations.pydantic_ai import GemmaDecisionModel


class Ticket(BaseModel):
    team: Literal["billing", "account", "technical"] = Field(
        description="Which support team should handle this ticket?"
    )
    urgent: bool = Field(description="Does the customer need attention today?")


async def main() -> None:
    async with GemmaDecisionModel() as model:
        agent = Agent(model, output_type=Ticket)
        result = await agent.run("My card was charged twice and I need a refund today.")
        print(result.output.model_dump_json(indent=2))
        print(result.response.provider_details)


if __name__ == "__main__":
    asyncio.run(main())
