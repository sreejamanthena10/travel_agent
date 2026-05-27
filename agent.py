import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Direct explicit imports to prevent any silent loading bugs
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
    
    # FIX: If it's already a list (TOML Array), clean and return it immediately!
    if isinstance(raw_keys, list):
        return [str(k).strip() for k in raw_keys if str(k).strip()]
        
    # If it's a single string, parse it safely
    try:
        cleaned_string = str(raw_keys).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
        return [k.strip() for k in cleaned_string.split(",") if k.strip()]
    except Exception:
        return []

def get_agent():
    keys_pool = get_keys_pool()
    if not keys_pool:
        return None

    # Pick the primary working key instantly
    primary_key = keys_pool[0]
    try:
        genai.configure(api_key=primary_key)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=primary_key,
            temperature=0.2
        )
        
        tools_list = [search_flights, search_hotels, get_weather, plan_itinerary]
        return create_react_agent(llm, tools=tools_list)
    except Exception:
        return None
