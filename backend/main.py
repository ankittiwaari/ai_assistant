import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from infra.client import get_client
from tools.web_search_tool import web_search

set_tracing_disabled(True)

agent = Agent(
    name="Main agent",
    instructions="You answer questions based on internet search clearly and concisely",
    tools=[web_search],
    model=OpenAIChatCompletionsModel(
        model="qwen3.5:4b",
        openai_client=get_client()
    ),    
)


async def main() -> None:
    result = await Runner.run(agent, "QUERY HERE")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())