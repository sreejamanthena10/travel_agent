import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# --- DYNAMIC STRUCTURAL IMPORT CONNECTOR ---
search_flights = None
search_hotels = None
get_weather = None
plan_itinerary = None

try:
    import tools
    import inspect
    functions_list = inspect.getmembers(tools, inspect.isfunction)
    for name, func in functions_list:
        name_lower = name.lower()
        if "flight" in name_lower and not search_flights: search_flights = func
        elif "hotel" in name_lower and not search_hotels: search_hotels = func
        elif "weather" in name_lower and not get_weather: get_weather = func
        elif ("itinerary" in name_lower or "plan" in name_lower) and not plan_itinerary: plan_itinerary = func
except Exception:
    pass

# Fallback assignments to strictly protect compilation execution state
if not search_flights:
    def search_flights(*args, **kwargs): return "Flight lookup tool currently unavailable."
if not search_hotels:
    def search_hotels(*args, **kwargs): return "Hotel information tool currently unavailable."
if not get_weather:
    def get_weather(*args, **kwargs): return "Weather telemetry matrix offline."
if not plan_itinerary:
    def plan_itinerary(*args, **kwargs): return "Itinerary generator routing offline."

def get_keys_pool():
    if "GEMINI_API_KEYS" not in st.secrets:
        return []
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

def get_agent():
    keys_pool = get_keys_pool()
    if not keys_pool:
        return None

    for current_key in keys_pool:
        try:
            genai.configure(api_key=current_key)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=current_key,
                temperature=0.2
            )
            
            tools_list = [search_flights, search_hotels, get_weather, plan_itinerary]
            compiled_agent = create_react_agent(llm, tools=tools_list)
            
            # Key status heartbeat test
            test_response = llm.invoke("Ping")
            if test_response and test_response.content:
                return compiled_agent
        except Exception:
            continue
    return None
