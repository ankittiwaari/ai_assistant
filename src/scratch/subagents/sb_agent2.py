import asyncio

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.entries import File
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

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
    name="Renewal Packet Analyst",
    model="gpt-6-astra",
    instructions=(
        "Review the workspace before answering. Keep the response concise, "
        "business-focused, and cite the file names that support each conclusion."
    ),
    default_manifest=manifest,
    capabilities=[Shell()],
)


async def main():
    result = await Runner.run(
        agent,
        "Summarize the renewal blockers and recommend the next two actions.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()),
            workflow_name="Unix-local sandbox review",
        ),
        max_turns=5,
    )
    print(result.final_output)


asyncio.run(main())
