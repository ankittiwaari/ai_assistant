import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from infra.client import get_client
from tools.web_search_tool import web_search

set_tracing_disabled(True)


web_search_agent = Agent(
    name="Web search agent",
    tools=[web_search],
    model=OpenAIChatCompletionsModel(
            model="qwen3.5:4b",
            openai_client=get_client()
        ),   
    handoff_description="Specialist for web search tasks.",
    instructions="You answer questions based on internet search clearly and concisely. Before answering, verify that you are answering exactly what user asked for. Never invent answers. When you don't know the answer, just say I don't know."
)

agent = Agent(
    name="Main agent",
    instructions="Route each task to the right specialist",
    handoffs = [web_search_agent],
    model=OpenAIChatCompletionsModel(
        model="qwen3.5:4b",
        openai_client=get_client()
    ),    
)


async def main() -> None:
    result = await Runner.run(agent, "Query here")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())