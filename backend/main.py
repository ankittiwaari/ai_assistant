import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from infra.client import get_client
from subagents.summariser_agent import summariser
from tools.web_search_tool import web_search
from tools.random_tool import latest_date
from dataclasses import dataclass
from datetime import date

set_tracing_disabled(True)

instructions:str = f"""Route each task to the right specialist. Whenever the invocation involves anything around latest, use the current date as {date.today()}."""

agent = Agent(
    name="Main agent",
    instructions=instructions,
    tools=[web_search, latest_date],
    handoffs = [summariser],
    model=OpenAIChatCompletionsModel(
        model="qwen3.5:4b",
        openai_client=get_client()
    ),    
)

@dataclass
class UserInfo:
    name:str
    uid:int

async def main() -> None:
    query = input("Enter your query:\n")
    result = await Runner.run(agent, query, context=UserInfo(name="John", uid=123))
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())