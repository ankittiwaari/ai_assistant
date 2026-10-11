from openai import AsyncOpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()


def get_client():
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=getenv("OPEN_ROUTER_API_KEY")
    )
