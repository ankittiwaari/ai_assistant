from openai import AsyncOpenAI


def get_client():
    return AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
