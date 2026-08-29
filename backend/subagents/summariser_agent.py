from agents import OpenAIChatCompletionsModel, Agent
from infra.client import get_client

summariser = Agent(
    name="Summariser agent",
    model=OpenAIChatCompletionsModel(
            model="qwen3.5:4b",
            openai_client=get_client()
        ),   
    handoff_description="Specialist for summarising.",
    instructions="You summarise the provided text into bullet points. The summary should never invent new information and never loose any important information."
)