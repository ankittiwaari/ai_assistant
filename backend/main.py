import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from infra.client import get_client
from subagents.summariser_agent import summariser
from tools.web_search_tool import web_search
from tools.random_tool import latest_date

set_tracing_disabled(True)

instructions:str = """Route each task to the right specialist. Whenever the invocation involves anything around latest, call the `latest_date` tool first to fetch today's date, then use that date to carry out the task."""

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

async def main() -> None:
    query = input("Enter your query:\n")
    result = await Runner.run(agent, query)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())