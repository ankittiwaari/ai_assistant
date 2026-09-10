import asyncio
from dataclasses import dataclass
from datetime import date

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

from infra.client import get_client
from infra.db import dispose_engine, get_session
from subagents.summariser_agent import summariser
from tools.random_tool import latest_date
from tools.web_search_tool import web_search

set_tracing_disabled(True)

instructions: str = f"""Route each task to the right specialist. Whenever the invocation involves anything around latest, use the current date as {date.today()}."""

agent = Agent(
    name="Main agent",
    instructions=instructions,
    tools=[web_search, latest_date],
    handoffs=[summariser],
    model=OpenAIChatCompletionsModel(model="qwen3.5:4b", openai_client=get_client()),
)


@dataclass
class UserInfo:
    name: str
    uid: int


async def init_chat(query: str, history, session_id: str = "user-123") -> str:
    print(history)
    result = await Runner.run(
        agent,
        query,
        context=UserInfo(name="John", uid=123),
        session=get_session(session_id),
    )
    print(result.final_output)
    return result.final_output


async def main():
    try:
        await init_chat(input("Enter your query"), [])
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
