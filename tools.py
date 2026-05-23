import requests
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Tool 1: Web Search (No API key needed)
web_search = DuckDuckGoSearchRun()

# Tool 2: Custom API with simple error handling
@tool
def get_weather(location: str) -> str:
    """Use this tool to find the current weather for a specific city or destination."""
    try:
        # We use wttr.in, a free text-based weather API
        response = requests.get(f"https://wttr.in/{location}?format=3", timeout=5)
        response.raise_for_status() # Checks for internet errors
        return f"Weather in {location}: {response.text}"
    except Exception as error:
        return f"Error getting weather data: {error}"

# Group the tools together
my_tools = [web_search, get_weather]
