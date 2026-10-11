from functools import cache
from os import getenv

from agents.extensions.memory import SQLAlchemySession
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

load_dotenv()


@cache
def get_engine() -> AsyncEngine:
    """Process-wide engine, so the connection pool is built once and reused.

    `create_async_engine` is a synchronous factory that only opens connections
    lazily, so there is nothing to await here.
    """
    user = getenv("DB_USER")
    password = getenv("DB_PASSWORD")
    host = getenv("DB_HOST")
    port = getenv("DB_PORT", "5432")
    name = getenv("DB_NAME")
    return create_async_engine(
        f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}",
        pool_pre_ping=True,
    )


def get_session(session_id: str) -> SQLAlchemySession:
    return SQLAlchemySession(session_id, engine=get_engine(), create_tables=True)


async def dispose_engine() -> None:
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        get_engine.cache_clear()
