import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

try:
    from tools import search_flights, search_hotels, get_weather, plan_itinerary
except ImportError:
    def search_flights(*args, **kwargs): return "Flights engine currently updating."
    def search_hotels(*args, **kwargs): return "Hotels database currently updating."
    def get_weather(*args, **kwargs): return "Weather telemetry system offline."
    def plan_itinerary(*args, **kwargs): return "Itinerary planner system offline."

# --- SYSTEM INSTRUCTION FOR BUDGET REASONING & OUTPUT STRUCTURE ---
SYSTEM_PROMPT = """You are a premium, highly actionable AI Travel Agent. 
When a user asks for a trip plan with a specific budget limit (e.g., under 50,000 or 4,500,000):
1. Never say you cannot factor in the budget. You MUST accept it and construct a plan tailored around it.
2. Call the required tools (search_flights, search_hotels, get_weather, plan_itinerary) to fetch raw market parameters.
3. Combine the tool outputs with your internal reasoning to create a clean, comprehensive markdown table titled 'Comprehensive Trip Expense Sheet (INR)'.
4. Break down costs for Flights, Hotels, Food, local transit, and miscellaneous matching the exact math requested. Ensure the total sits safely under the budget cap.
5. Provide a clear, scannable day-by-day sightseeing layout and print out the live weather telemetry data directly on screen. Do not output raw JSON, signatures, or error text blocks."""

def get_keys_pool():
    if "GEMINI_API_KEYS" not in st.secrets:
        return []
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    
    if isinstance(raw_keys, str):
        if "," in raw_keys:
            return [k.strip() for k in raw_keys.split(",") if k.strip()]
        return [raw_keys.strip()]
        
    if isinstance(raw_keys, list):
        return [str(k).strip() for k in raw_keys if str(k).strip()]
        
    return []

def get_agent():
    keys_pool = get_keys_pool()
    if not keys_pool:
        return None
        
    tools_list = [search_flights, search_hotels, get_weather, plan_itinerary]
    
    # Grab the first key in your pool directly without burning rate limits on a test ping
    for active_key in keys_pool:
        try:
            clean_key = active_key.replace("[", "").replace("]", "").replace('"', '').replace("'", "").strip()
            
            genai.configure(api_key=clean_key)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=clean_key,
                temperature=0.0,
                max_retries=3
            )
            
            # --- FIXED: TEST PING REMOVED TO PREVENT COLD-START 429 RATE LIMIT BLOCKS ---
            return create_react_agent(llm, tools=tools_list, state_modifier=SYSTEM_PROMPT)
        except Exception:
            continue
            
    return None
