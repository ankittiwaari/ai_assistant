from agents import Agent, OpenAIChatCompletionsModel

from app.infra.client import get_client

summariser = Agent(
    name="Summariser agent",
    model=OpenAIChatCompletionsModel(
        model="nvidia/nemotron-3-ultra-550b-a55b:free", openai_client=get_client()
    ),
    handoff_description="Specialist for summarising.",
    instructions="You summarise the provided text into bullet points. The summary should never invent new information and never loose any important information.",
)
