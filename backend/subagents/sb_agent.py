import asyncio

from agents import OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.run import RunConfig

from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Capabilities
from infra.client import get_client
from infra.db import dispose_engine

agent = SandboxAgent(
    model=OpenAIChatCompletionsModel(model="qwen3.5:4b", openai_client=get_client()),
    name="Dummy sandbox agent",
    instructions="Use the mounted skills before formulating response",
    capabilities=Capabilities.default()
)

set_tracing_disabled(True)

async def init_chat(query: str, history, session_id: str = "user-123") -> str:
    result = await Runner.run(
        agent,
        query,
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()),
            workflow_name="Unix-local sandbox review",
        ),
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