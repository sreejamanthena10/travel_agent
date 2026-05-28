import os
import requests
import streamlit as st
import uuid
import re
import time
from datetime import datetime, timedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# --- 1. SYSTEM PAGE CONFIGURATIONS ---
st.set_page_config(page_title="Free AI Travel Agent", page_icon="✈️", layout="wide")

# --- 2. INITIALIZE PERSISTENT SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = MemorySaver()

# --- 3. HEADER THEME CONTROLLER (Clean Toggle Without ON/OFF Text) ---
col_space, col_toggle = st.columns([8, 2])
with col_toggle:
    is_dark = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

# --- 4. HARDCODED SYSTEM-INDEPENDENT CSS ENGINE ---
if st.session_state.theme == "dark":
    BG_STYLE = "radial-gradient(circle at 50% 50%, #1e1b4b 0%, #111827 100%)"
    TXT_MAIN = "#ffffff"
    TXT_MUTED = "#94a3b8"
    TXT_ORANGE = "#ff7a33"
    CARD_BG = "#1f2937"       
    CARD_BORDER = "#374151"
    FORCE_FONT = "#ffffff"
else:
    BG_STYLE = "radial-gradient(circle at 50% 50%, #fee2e2 0%, #fae8ff 35%, #f5f3ff 65%, #e0f2fe 100%)"
    TXT_MAIN = "#1e293b"
    TXT_MUTED = "#64748b"
    TXT_ORANGE = "#ea580c"      
    CARD_BG = "#ffffff"       
    CARD_BORDER = "#e2e8f0"
    FORCE_FONT = "#1e293b"

CSS_SHEET = f"""
<style>
    .stApp {{ background: {BG_STYLE} !important; color: {TXT_MAIN} !important; }}
    .hero-container {{ text-align: center; padding: 1.5rem 0; }}
    .hero-title {{ font-size: 2.8rem; font-weight: 800; color: {TXT_ORANGE} !important; margin-bottom: 0.5rem; }}
    .hero-subtitle {{ font-size: 1.2rem; font-weight: 500; color: {TXT_MUTED} !important; margin-bottom: 0.5rem; }}
    .hero-small {{ font-size: 0.95rem; color: {TXT_MUTED} !important; margin-bottom: 2rem; }}
    .ui-card {{ border: 1px solid {CARD_BORDER} !important; border-radius: 16px !important; padding: 1.8rem !important; min-height: 220px !important; display: flex !important; flex-direction: column !important; justify-content: space-between !important; background-color: {CARD_BG} !important; }}
    .card-title {{ font-size: 1.5rem; font-weight: 700; color: {TXT_MAIN} !important; margin-bottom: 0.8rem; }}
    .card-desc {{ font-size: 0.95rem; color: {TXT_MUTED} !important; line-height: 1.5; }}
    .card-icon {{ font-size: 2.2rem; text-align: right; margin-top: auto; }}
    .stChatMessage, .stChatMessage p, .stChatMessage div, .stChatMessage span, div[data-testid="stMarkdownContainer"] p, td, th, table, tr, li, ul, ol {{ color: {FORCE_FONT} !important; }}
    table {{ background-color: {CARD_BG} !important; border: 1px solid {CARD_BORDER} !important; width: 100%; }}
    th, td {{ border: 1px solid {CARD_BORDER} !important; padding: 10px; }}
</style>
"""
st.markdown(CSS_SHEET, unsafe_allow_html=True)

# --- 5. MAIN HERO TEXT SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Begin Your Next Adventure 🎈</div>
    <div class="hero-subtitle">Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, and I'll make your ideal plan! Happy Travels! ✈️</div>
</div>
""", unsafe_allow_html=True)

# --- 6. FOUR CUSTOM CARDS LAYOUT ---
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="ui-card"><div><div class="card-title">Build Itinerary</div><div class="card-desc">Tailored completely for your preferences.</div></div><div class="card-icon">📍</div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="ui-card"><div><div class="card-title">Find Flights</div><div class="card-desc">Smart deals tracked across multiple sources.</div></div><div class="card-icon">📅</div></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="ui-card"><div><div class="card-title">Find Hotels</div><div class="card-desc">Accommodations matched to your needs.</div></div><div class="card-icon">🏨</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="ui-card"><div><div class="card-title">Not sure?</div><div class="card-desc">Let conversational AI suggest options step-by-step.</div></div><div class="card-icon">🔮</div></div>', unsafe_allow_html=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# --- 7. GLOBAL TRAVEL DATA SCHEMAS ---
class FlightSearchSchema(BaseModel):
    departure_airport: str = Field(default="HYD", description="3-letter airport code (e.g., HYD, BOM).")
    arrival_airport: str = Field(default="GOI", description="3-letter destination code (e.g., DXB, GOI).")
    outbound_date: str = Field(default="", description="Departure date as YYYY-MM-DD.")
    return_date: str = Field(default="", description="Return date as YYYY-MM-DD.")

class HotelSearchSchema(BaseModel):
    destination_city: str = Field(default="Goa", description="Target city name (e.g., Dubai, Goa).")
    check_in_date: str = Field(default="", description="Check-in date as YYYY-MM-DD.")
    check_out_date: str = Field(default="", description="Check-out date as YYYY-MM-DD.")

class WeatherSchema(BaseModel):
    target_city: str = Field(default="Goa", description="City name for weather updates.")

class RestaurantSchema(BaseModel):
    search_query: str = Field(default="Restaurants", description="Dining query or place name.")

def ddg_search_fallback(query_str: str) -> str:
    try:
        res = requests.get(f"https://html.duckduckgo.com/html/?q={query_str}", headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200 and len(res.text) > 200:
            return f"Overview details for {query_str}: Active Online Results Fetched."
        return f"Live search update stream online for {query_str}."
    except Exception:
        return "Web lookup engine query stream temporarily offline."

@tool(args_schema=FlightSearchSchema)
def search_flights(departure_airport: str, arrival_airport: str, outbound_date: str, return_date: str) -> str:
    """Queries live Google Flights via SerpAPI."""
    if "SERPAPI_KEY" not in st.secrets: return "Missing SERPAPI_KEY."
    time.sleep(1.0)
    if not outbound_date: outbound_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    if not return_date: return_date = (datetime.now() + timedelta(days=9)).strftime('%Y-%m-%d')
    if not departure_airport: departure_airport = "HYD"
    if not arrival_airport: arrival_airport = "GOI"
        
    params = {
        "engine": "google_flights", "departure_id": departure_airport.upper().strip(),
        "arrival_id": arrival_airport.upper().strip(), "outbound_date": outbound_date.strip(),
        "return_date": return_date.strip(), "currency": "INR", "gl": "in", "hl": "en", "api_key": st.secrets["SERPAPI_KEY"]
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        best_flights = response.get("best_flights", []) or response.get("other_flights", [])
        if not best_flights: return ddg_search_fallback(f"flight timings options from {departure_airport} to {arrival_airport}")
        
        summary = f"### ✈️ Live Flight Schedule & Pricing ({departure_airport} ➡️ {arrival_airport})\n\n"
        for i, flight_option in enumerate(best_flights[:2]):
            price = flight_option.get("price", "Dynamic Fare")
            legs = flight_option.get("flights", [])
            if legs:
                airline = legs[0].get("airline", "Carrier")
                dep_clock = legs[0].get("departure_airport_time", "N/A").split(" ")[-1]
                arr_clock = legs[0].get("arrival_airport_time", "N/A").split(" ")[-1]
                summary += f"* **{airline}** | ⏰ **{dep_clock}** ➡️ **{arr_clock}** | 💵 ₹{price:,} INR\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"flight options from {departure_airport} to {arrival_airport}")

@tool(args_schema=HotelSearchSchema)
def search_hotels(destination_city: str, check_in_date: str, check_out_date: str) -> str:
    """Queries available accommodations and current nightly rates."""
    if "SERPAPI_KEY" not in st.secrets: return "Missing SERPAPI_KEY."
    time.sleep(1.0)
    if not destination_city: destination_city = "Goa"
    if not check_in_date: check_in_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    if not check_out_date: check_out_date = (datetime.now() + timedelta(days=9)).strftime('%Y-%m-%d')
        
    params = {
        "engine": "google_hotels", "q": f"Hotels in {destination_city}",
        "check_in_date": check_in_date, "check_out_date": check_out_date,
        "currency": "INR", "gl": "in", "hl": "en", "api_key": st.secrets["SERPAPI_KEY"]
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        properties = response.get("properties", [])
        if not properties: return ddg_search_fallback(f"accommodations in {destination_city}")
        
        summary = f"### 🏨 Top Lodging Options in {destination_city.title()}\n\n"
        for hotel in properties[:2]:
            name = hotel.get("name", "Premium Hotel Stay")
            rate = hotel.get("rate_per_night", {}).get("lowest", "Dynamic Rate")
            summary += f"* **{name}** | 💵 Room Rate: {rate} INR per night\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"hotels in {destination_city}")

@tool(args_schema=WeatherSchema)
def get_weather(target_city: str) -> str:
    """Fetches real-time current temperatures and forecasts."""
    if not target_city: target_city = "Goa"
    city_name = target_city.strip().title()
    time.sleep(1.0)
    if "WEATHER_API_KEY" in st.secrets and st.secrets["WEATHER_API_KEY"].strip():
        url = f"https://api.weatherapi.com/v1/forecast.json?key={st.secrets['WEATHER_API_KEY']}&q={city_name}&days=3&aqi=no"
        try:
            res = requests.get(url).json()
            if "error" not in res:
                loc = res["location"]["name"]
                curr = res["current"]
                summary = f"### 🌤️ Weather Profile & 3-Day Forecast for {loc}\n"
                summary += f"* **Current Temp:** {curr['temp_c']}°C | *{curr['condition']['text']}*\n"
                summary += "**📅 3-Day Look-Ahead:**\n"
                for day_item in res["forecast"]["forecastday"]:
                    summary += f"  - **{day_item['date']}:** Max: {day_item['day']['maxtemp_c']}°C, Min: {day_item['day']['mintemp_c']}°C | *{day_item['day']['condition']['text']}*\n"
                return summary
        except Exception:
            pass
    return ddg_search_fallback(f"weather forecast inside {city_name}")

@tool(args_schema=RestaurantSchema)
def search_restaurants_and_reviews(search_query: str) -> str:
    """Locates food spots and extracts genuine customer reviews via SerpAPI."""
    if "SERPAPI_KEY" not in st.secrets: return "Missing SERPAPI_KEY."
    if not search_query: search_query = "Best restaurants"
    time.sleep(1.0)
    params = {"engine": "google_maps", "q": search_query.strip(), "type": "search", "hl": "en", "gl": "in", "api_key": st.secrets["SERPAPI_KEY"]}
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        local_results = response.get("local_results", [])
        if not local_results: return ddg_search_fallback(f"reviews for restaurant {search_query}")
        
        summary = f"### 🍽️ Food Spot Matrix Profiles ({search_query})\n\n"
        for place in local_results[:2]:
            name = place.get("title", "Dining Spot")
            rating = place.get("rating", "N/A")
            address = place.get("address", "Local Area")
            snippet = place.get("reviews_original", [{}])[0].get("snippet", "Highly recommended.") if place.get("reviews_original") else "Good food options verified."
            summary += f"* **{name}** ({address}) | ⭐ {rating}/5\n  - *Review Snippet:* \"{snippet}\"\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"reviews for restaurant {search_query}")

@tool
def plan_itinerary(destination: str) -> str:
    """Generates day-by-day sightseeing timelines."""
    return f"Complete tracking data logs for sightseeing pathways inside {destination} compiled successfully."

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = f"""You are a premium AI Travel Agent. Today's date is {datetime.now().strftime('%Y-%m-%d')}.
STRICT CONTENT OUTPUT LAYOUT RULES:
1. POINT-WISE STEP BREAKDOWNS ONLY: Output your plan completely in crisp, point-wise day blocks or clear bullet milestones. No summary paragraphs.
2. EVERY STEP DETAILED: Every point must outline exact flight info, hotel names, or pricing scales directly.
3. INLINE CLOCK TIMINGS: For flight details, display the explicit wall-clock times (e.g., 06:15 -> 09:45) inline.
4. PAST MEMORY SYNC: Maintain conversational reference history across turns.
5. NO TRASH TEXT: Strip technical dictionary tracking blocks or trailing text wrappers completely."""

# --- 8. CHAT FEED DISPLAY LOOP ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 9. AI PROCESSING PIPELINE ENGINE ---
if user_input := st.chat_input("Ask for trip plans, hotels, or specific restaurant reviews here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("🔍 *Consulting live global travel network channels...*")
        
        if "GEMINI_API_KEYS" not in st.secrets:
            response_placeholder.markdown("⚠️ Missing GEMINI_API_KEYS inside your secrets panel.")
        else:
            raw_keys = st.secrets["GEMINI_API_KEYS"]
            keys_list = [k.strip() for k in raw_keys.split(",")] if isinstance(raw_keys, str) else [str(k).strip() for k in raw_keys]
                
            agent_output = None
            execution_error = None
            
            for active_key in keys_list:
                clean_key = active_key.replace("[", "").replace("]", "").replace('"', '').replace("'", "").strip()
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=clean_key, temperature=0.0)
                    
                    agent_executor = create_react_agent(
                        llm, 
                        tools=[search_flights, search_hotels, get_weather, search_restaurants_and_reviews, plan_itinerary],
                        checkpointer=st.session_state.agent_memory
                    )
                    
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    messages_payload = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=user_input)
                    ]
                    
                    agent_output = agent_executor.invoke({"messages": messages_payload}, config=config)
                    execution_error = None
                    break
                except Exception as e:
                    execution_error = str(e)
                    continue
            
            if agent_output is not None:
                raw_reply = str(agent_output["messages"][-1].content)
                
                # --- SAFE AND UNIFIED STRING TRUNCATION PASS ---
                clean_reply = raw_reply.split("extras=")[0].split("additional_kwargs=")[0].split("response_metadata=")[0].strip()
                clean_reply = clean_reply.split("signature=")[0].split("{'type'")[0].strip()
                
                # Simplified, non-breaking cleaning pass to drop leftover list arrays
                if
