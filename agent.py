import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Direct explicit tool imports
try:
    from tools import search_flights, search_hotels, get_weather, plan_itinerary
except ImportError:
    def search_flights(*args, **kwargs): return "Flights engine currently updating."
    def search_hotels(*args, **kwargs): return "Hotels database currently updating."
    def get_weather(*args, **kwargs): return "Weather telemetry system offline."
    def plan_itinerary(*args, **kwargs): return "Itinerary planner system offline."

def get_keys_pool():
    if "GEMINI_API_KEYS" not in st.secrets:
        return []
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    if isinstance(raw_keys, list):
        return [str(k).strip() for k in raw_keys if str(k).strip()]
    try:
        cleaned_string = str(raw_keys).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
        return [k.strip() for k in cleaned_string.split(",") if k.strip()]
    except Exception:
        return []

def get_agent():
    keys_pool = get_keys_pool()
    if not keys_pool:
        return None

    tools_list = [search_flights, search_hotels, get_weather, plan_itinerary]

    # Dynamically build and test the agent against the active keys pool
    for active_key in keys_pool:
        try:
            genai.configure(api_key=active_key)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=active_key,
                temperature=0.1,  # Lower temperature = faster, more focused tool calling
                max_retries=1    # Stops the agent from wasting time retrying a dead key
            )
            
            # Re-compile state graph for this key instance
            agent_executor = create_react_agent(llm, tools=tools_list)
            return agent_executor
        except Exception:
            continue # If this key has an issue, immediately skip to the next one
            
    return None
