import asyncio
from os import getenv

import docker
from agents import OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Memory, Shell
from agents.sandbox.entries import File
from agents.sandbox.sandboxes import DockerSandboxClient, DockerSandboxClientOptions
from dotenv import load_dotenv

from src.app.infra.client import get_client
from src.app.infra.db import dispose_engine

load_dotenv()


manifest = Manifest(
    entries={
        "account_brief.md": File(
            content=(
                b"# Northwind Health\n\n"
                b"- Segment: Mid-market healthcare analytics provider.\n"
                b"- Renewal date: 2026-04-15.\n"
            )
        ),
        "implementation_risks.md": File(
            content=(
                b"# Delivery risks\n\n"
                b"- Security questionnaire is not complete.\n"
                b"- Procurement requires final legal language by April 1.\n"
            )
        ),
    }
)

agent = SandboxAgent(
    model=OpenAIChatCompletionsModel(
        model=getenv("OPEN_ROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        openai_client=get_client(),
    ),
    name="Dummy sandbox agent",
    instructions=(
        "Review the workspace before answering. Keep the response concise, "
        "business-focused, and cite the file names that support each conclusion."
    ),
    default_manifest=manifest,
    capabilities=[Shell(), Memory(), Filesystem()],
)

set_tracing_disabled(True)


async def init_chat(query: str, history, session_id: str = "user-123") -> str:
    # result = await Runner.run(
    #     agent,
    #     query,
    #     run_config=RunConfig(
    #         sandbox=SandboxRunConfig(
    #             client=DockerSandboxClient(docker.from_env()),
    #             options=DockerSandboxClientOptions(image="python:3.13-slim"),
    #         ),
    #         workflow_name="Docker sandbox review",
    #     ),
    # )
    # print(result.final_output)
    # return result.final_output

    result = await Runner.run(
        agent,
        "Summarize the renewal blockers and recommend the next two actions.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=DockerSandboxClient(docker.from_env()),
                options=DockerSandboxClientOptions(image="python:3.13-slim"),
            ),
            workflow_name="Unix-local sandbox review",
        ),
    )
    print(result.final_output)


async def main():
    try:
        await init_chat("", [])
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
