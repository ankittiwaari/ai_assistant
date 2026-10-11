from asyncio import run
from os import getenv

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from pydantic import BaseModel

from src.app.infra.client import get_client

load_dotenv()

set_tracing_disabled(True)


class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str


@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_homework,
    )


guardrail_agent = Agent(
    name="Homework checker",
    instructions="Detect whether the user is asking for maths homework help.",
    output_type=MathHomeworkOutput,
    model=OpenAIChatCompletionsModel(
        model=getenv("OPEN_ROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        openai_client=get_client(),
    ),
)

main_agent = Agent(
    name="Assistant",
    instructions="Help the user with their questions.",
    model=OpenAIChatCompletionsModel(
        model=getenv("OPEN_ROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        openai_client=get_client(),
    ),
    input_guardrails=[math_guardrail],
)


async def main() -> None:
    try:
        result = await Runner.run(main_agent, "Can you write a oneliner horror story?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print("Guardrail blocked the request.")


if __name__ == "__main__":
    run(main())
