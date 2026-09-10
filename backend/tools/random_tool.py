from datetime import date

from agents import function_tool


@function_tool
def latest_date() -> date:
    """Return today's date. Use this tool whenever user's query asks for anything latest."""
    return date.today()
