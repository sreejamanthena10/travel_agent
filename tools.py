import requests
import json
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# --- Tool 1: Clean Custom Search Rewrite ---
@tool
def my_search_tool(query: str) -> str:
    """Use this tool to search the web for current events, facts, or up-to-date travel info."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No matching search results found on the web."
            
            summary = []
            for r in results:
                summary.append(f"Title: {r.get('title')}\nSource: {r.get('body')}\n")
            return "\n".join(summary)
            
    except Exception as e:
        return f"Could not complete web search: {str(e)}"

# --- Tool 2: Safe Weather API with JSON Telemetry Stripping ---
@tool
def get_weather(location: str) -> str:
    """Use this tool to find the current weather for a specific city or destination."""
    try:
        # Requesting plain-text format from wttr.in
        response = requests.get(f"https://wttr.in/{location}?format=3", timeout=5)
        response.raise_for_status()
        raw_text = response.text.strip()
        
        # SAFETY CHECK: If the response accidentally contains a raw json block, extract just the text
        if raw_text.startswith("{") or '"text"' in raw_text:
            try:
                # Clean edge cases where raw strings resemble arrays
                clean_json_str = raw_text.lstrip("0:").strip("[] \n")
                data = json.loads(clean_json_str)
                return f"Weather in {location}: {data.get('text', 'No description available')}"
            except Exception:
                # Fallback if manual parsing trips up on nested tokens
                if '"text":"' in raw_text:
                    extracted = raw_text.split('"text":"')[1].split('","extras"')[0]
                    return f"Weather in {location}: {extracted}"
        
        return f"Weather in {location}: {raw_text}"
        
    except Exception as error:
        return f"Error getting weather data: {error}"

# Group the tools together cleanly
my_tools = [my_search_tool, get_weather]
