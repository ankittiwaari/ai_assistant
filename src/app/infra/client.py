from os import getenv

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def get_client():
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=getenv("OPEN_ROUTER_API_KEY")
    )
