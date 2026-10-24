import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from os import getenv

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv

from app.infra.client import get_client
from app.infra.db import dispose_engine, get_session
from app.infra.logging_config import setup_logging
from app.rag_agent.retrieval.retrieve_context import search_documents
from app.tools.random_tool import latest_date, logger_function
from app.tools.web_search_tool import web_search
from scratch.subagents.summariser_agent import summariser

load_dotenv()


set_tracing_disabled(True)

instructions: str = f"""Route each task to the right specialist. Whenever the invocation involves anything around latest, use the current date as {datetime.now(tz=UTC).date()}."""

agent = Agent(
    name="Main agent",
    instructions=instructions
    + (
        " For questions about the documents, call search_documents and answer "
        "only from the returned passages, citing page numbers. If the passages "
        "don't contain the answer, say you don't know."
    ),
    tools=[web_search, latest_date, search_documents, logger_function],
    handoffs=[summariser],
    model=OpenAIChatCompletionsModel(
        model=getenv("OPEN_ROUTER_MODEL_PAID", "google/gemma-4-26b-a4b-it:free"),
        openai_client=get_client(),
    ),
)


@dataclass
class UserInfo:
    name: str
    uid: int


async def init_chat(query: str, history, session_id: str = "user-123") -> str:
    result = await Runner.run(
        agent,
        query,
        context=UserInfo(name="John", uid=123),
        session=get_session(session_id),
    )
    return result.final_output


async def run():
    setup_logging()
    try:
        response = await init_chat(input("Enter your query:\n"), [])
        print(response)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)
