import asyncio
from os import getenv

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from dotenv import load_dotenv

from src.app.infra.client import get_client

load_dotenv()

model = OpenAIChatCompletionsModel(
    model=getenv("OPEN_ROUTER_MODEL", ""), openai_client=get_client()
)


@function_tool(needs_approval=True)
async def cancel_order(order_id: int):
    return f"Cancelled order id {order_id}"


agent = Agent(
    name="Customer support agent",
    instructions="Handle support questions and seek approval when needed",
    tools=[cancel_order],
    model=model,
)


async def main() -> None:
    result = await Runner.run(agent, "cancel order 123")
    if result.interruptions:
        state = result.to_state()
        for intr in result.interruptions:
            user_input = input(f"Approve interruption {intr}? Y/N")
            if user_input == "Y" or user_input == "y":
                state.approve(intr)
            else:
                state.reject(intr)
        result = await Runner.run(agent, state)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
