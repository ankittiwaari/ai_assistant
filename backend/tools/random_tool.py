from agents import function_tool

@function_tool
def fun_fact() -> str:
    """Return a dummy fun fact"""
    return "just a random dummy text"