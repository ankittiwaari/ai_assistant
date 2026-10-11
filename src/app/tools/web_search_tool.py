from os import getenv
from typing import Any

from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient(getenv("TAVILY_API_KEY"))


@function_tool
def web_search(wrapper: RunContextWrapper, query: str) -> dict[str, Any]:
    """Use this tool whenever you need to search the web"""
    print(f"Using websearch with: {query}")
    print(f"User: {wrapper.context.name}")
    return client.search(query=query, search_depth="advanced")
