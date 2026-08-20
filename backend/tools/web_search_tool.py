from tavily import TavilyClient
from dotenv import load_dotenv
from os import getenv
from agents import function_tool
from typing import Any

load_dotenv()
client = TavilyClient(getenv('TAVILY_API_KEY'))

@function_tool
def web_search(query: str) -> dict[str, Any]:
    """Use this tool whenever you need to search the web"""
    print(f"Using websearch with: {query}")
    return client.search(
        query=query,
        search_depth="advanced"
    )