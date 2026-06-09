from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

@tool
def web_search(query: str):
    """
    Performs a web search to find recent news and information.
    Args:
        query: The search query.
    """
    search = DuckDuckGoSearchRun()
    return search.invoke(query)
