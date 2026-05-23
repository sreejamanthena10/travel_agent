import requests
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# Tool 1: Clean custom rewrite of the Web Search Tool using raw python
@tool
def my_search_tool(query: str) -> str:
    """Use this tool to search the web for current events, facts, or up-to-date travel info."""
    try:
        # Calls the search engine library directly, bypassing the broken LangChain validator
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No matching search results found on the web."
            
            # Combine the titles and snippets nicely
            summary = []
            for r in results:
                summary.append(f"Title: {r.get('title')}\nSource: {r.get('body')}\n")
            return "\n".join(summary)
            
    except Exception as e:
        return f"Could not complete web search due to an error: {str(e)}"

# Tool 2: Custom API with simple error handling
@tool
def get_weather(location: str) -> str:
    """Use this tool to find the current weather for a specific city or destination."""
    try:
        response = requests.get(f"https://wttr.in/{location}?format=3", timeout=5)
        response.raise_for_status()
        return f"Weather in {location}: {response.text}"
    except Exception as error:
        return f"Error getting weather data: {error}"

# Group the tools together exactly like before
my_tools = [my_search_tool, get_weather]
