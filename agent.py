import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

try:
    from tools import search_flights, search_hotels, get_weather, plan_itinerary
except ImportError:
    def search_flights(*args, **kwargs): return "Flights engine updating."
    def search_hotels(*args, **kwargs): return "Hotels database updating."
    def get_weather(*args, **kwargs): return "Weather telemetry offline."
    def plan_itinerary(*args, **kwargs): return "Itinerary planner offline."

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

    for active_key in keys_pool:
        try:
            genai.configure(api_key=active_key)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=active_key,
                temperature=0.1,
                max_retries=3  # Smart choice: Automatically bypasses temporary 503 traffic spikes in milliseconds
            )
            
            # Fast verification test
            llm.invoke("ping")
            
            return create_react_agent(llm, tools=tools_list)
        except Exception:
            continue  # Seamlessly skips to the next key if a key is hard-blocked or exhausted
            
    return None
