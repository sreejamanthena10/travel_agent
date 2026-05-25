import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# --- DYNAMIC UNIVERSAL INSPECTION SCANNER GATEWAY ---
# Automatically matches your tools.py functions by scanning for keywords
search_flights = None
search_hotels = None
get_weather = None
plan_itinerary = None

try:
    import tools
    import inspect
    
    # Get a list of all functions defined explicitly inside your tools.py
    functions_list = inspect.getmembers(tools, inspect.isfunction)
    
    for name, func in functions_list:
        name_lower = name.lower()
        
        # Dynamic keyword pattern matching flags
        if "flight" in name_lower and not search_flights:
            search_flights = func
        elif "hotel" in name_lower and not search_hotels:
            search_hotels = func
        elif ("weather" in name_lower or "temp" in name_lower or "climate" in name_lower) and not get_weather:
            get_weather = func
        elif ("itinerary" in name_lower or "plan" in name_lower or "suggest" in name_lower) and not plan_itinerary:
            plan_itinerary = func

    # Validation Guardrails: Ensure all 4 vectors found a matching functional code block
    if not search_flights:
        raise ImportError("Could not dynamically resolve your Flight tool function name inside tools.py")
    if not search_hotels:
        raise ImportError("Could not dynamically resolve your Hotel tool function name inside tools.py")
    if not get_weather:
        raise ImportError("Could not dynamically resolve your Weather tool function name inside tools.py")
    if not plan_itinerary:
        raise ImportError("Could not dynamically resolve your Itinerary tool function name inside tools.py")

except ImportError as e:
    st.error(f"❌ Structural Alignment Mapping Failure: {str(e)}")
    # Fallback blank definitions to allow interface compilation to clear during hot reloads
    def search_flights(*args, **kwargs): return "Flight tool offline"
    def search_hotels(*args, **kwargs): return "Hotel tool offline"
    def get_weather(*args, **kwargs): return "Weather tool offline"
    def plan_itinerary(*args, **kwargs): return "Itinerary tool offline"

def get_keys_pool():
    """Safely extracts and parses the comma-separated key pool string from Streamlit Secrets."""
    if "GEMINI_API_KEYS" not in st.secrets:
        return []
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

def get_agent():
    """Cycles through independent project keys to compile a warm, stable live agent loop."""
    keys_pool = get_keys_pool()
    
    if not keys_pool:
        print("❌ Config Error: No structural token strings registered inside secrets pool.")
        return None

    # Loop dynamically through each distinct project key location
    for current_key in keys_pool:
        try:
            # Configure core generative bindings
            genai.configure(api_key=current_key)
            
            # Initialize model context using the active key index loop pass
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=current_key,
                temperature=0.3
            )
            
            # Assemble your safe, mapped tool arrays
            tools_list = [search_flights, search_hotels, get_weather, plan_itinerary]
            
            # Compile the react graph state machine
            compiled_react_agent = create_react_agent(llm, tools=tools_list)
            
            # Lightweight validation request to ensure project token has clear remaining quota limits
            test_response = llm.invoke("Ping")
            
            if test_response and test_response.content:
                # Active healthy link established! Return agent context stream immediately
                return compiled_react_agent
                
        except Exception as e:
            error_msg = str(e).lower()
            # If the current key drops a 429 quota block, skip to the next project entry block instantly
            if "429" in error_msg or "resource_exhausted" in error_msg or "invalid" in error_msg or "expired" in error_msg:
                continue
            else:
                continue

    return None
