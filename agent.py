import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# --- SAFE ALIASED IMPORT GATEWAY ---
# This block safely inspects tools.py and loads functions regardless of slight naming differences
try:
    import tools
    
    # 1. Resolve Flight Tool Name Variance
    if hasattr(tools, "search_flights"): search_flights = tools.search_flights
    elif hasattr(tools, "flight_search"): search_flights = tools.flight_search
    elif hasattr(tools, "search_flight"): search_flights = tools.search_flight
    else: raise ImportError("Flight search tool function matching failed inside tools.py")

    # 2. Resolve Hotel Tool Name Variance
    if hasattr(tools, "search_hotels"): search_hotels = tools.search_hotels
    elif hasattr(tools, "hotel_search"): search_hotels = tools.hotel_search
    elif hasattr(tools, "search_hotel"): search_hotels = tools.search_hotel
    else: raise ImportError("Hotel search tool function matching failed inside tools.py")

    # 3. Resolve Weather Tool Name Variance
    if hasattr(tools, "get_weather"): get_weather = tools.get_weather
    elif hasattr(tools, "weather_search"): get_weather = tools.weather_search
    elif hasattr(tools, "fetch_weather"): get_weather = tools.fetch_weather
    else: raise ImportError("Weather tool function matching failed inside tools.py")

    # 4. Resolve Itinerary Tool Name Variance
    if hasattr(tools, "plan_itinerary"): plan_itinerary = tools.plan_itinerary
    elif hasattr(tools, "itinerary_planner"): plan_itinerary = tools.itinerary_planner
    elif hasattr(tools, "create_itinerary"): plan_itinerary = tools.create_itinerary
    else: raise ImportError("Itinerary tool function matching failed inside tools.py")

except ImportError as e:
    st.error(f"❌ Structural Alignment Mapping Failure: {str(e)}")
    # Fallback placeholders to allow compilation to proceed safely during alignment checks
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
