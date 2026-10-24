from datetime import date, datetime
from zoneinfo import ZoneInfo

from agents import function_tool


@function_tool
def latest_date() -> date:
    """Return today's date. Use this tool whenever user's query asks for anything latest."""
    return datetime.now(tz=ZoneInfo("Asia/Kolkata")).date()


@function_tool
def logger_function() -> str:
    """
    Always call this tool, no matter what the query is.
    """
    print("MANDATORY FN CALLED")
    return "Mandatory tool call"
