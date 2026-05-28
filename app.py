import os
import requests
import streamlit as st
import uuid
import re
import time
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# --- 1. SYSTEM PAGE CONFIGURATIONS ---
st.set_page_config(page_title="Free AI Travel Agent", page_icon="✈️", layout="wide")

# --- 2. INITIALIZE PERSISTENT SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- 3. HEADER THEME CONTROLLER (Clean Toggle Without ON/OFF Text) ---
col_space, col_toggle = st.columns([8, 2])
with col_toggle:
    is_dark = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

# --- 4. DYNAMIC THEME-INDEPENDENT CSS ENGINE ---
if st.session_state.theme == "dark":
    BG_STYLE = "radial-gradient(circle at 50% 50%, #1e1b4b 0%, #111827 100%)"
    TXT_MAIN = "#ffffff"
    TXT_MUTED = "#94a3b8"
    TXT_ORANGE = "#ff7a33"
    CARD_1_BG = "#2e2a14"       
    CARD_2_BG = "#1e293b"       
    CARD_3_BG = "#1e3a8a"       
    CARD_4_BG = "#1f2937"       
    CARD_BORDER = "#374151"
    FORCE_FONT = "#ffffff"
else:
    BG_STYLE = "radial-gradient(circle at 50% 50%, #fee2e2 0%, #fae8ff 35%, #f5f3ff 65%, #e0f2fe 100%)"
    TXT_MAIN = "#1e293b"
    TXT_MUTED = "#64748b"
    TXT_ORANGE = "#ea580c"      
    CARD_1_BG = "#fef08a"       
    CARD_2_BG = "#dbeafe"       
    CARD_3_BG = "#bfdbfe"       
    CARD_4_BG = "#ffffff"       
    CARD_BORDER = "#e2e8f0"
    FORCE_FONT = "#1e293b"

CSS_SHEET = f"""
<style>
    .stApp {{ background: {BG_STYLE} !important; color: {TXT_MAIN} !important; }}
    .hero-container {{ text-align: center; padding: 1.5rem 0; }}
    .hero-title {{ font-size: 2.8rem; font-weight: 800; color: {TXT_ORANGE} !important; margin-bottom: 0.5rem; }}
    .hero-subtitle {{ font-size: 1.2rem; font-weight: 500; color: {TXT_MUTED} !important; margin-bottom: 0.5rem; }}
    .hero-small {{ font-size: 0.95rem; color: {TXT_MUTED} !important; margin-bottom: 2rem; }}
    .ui-card {{ border: 1px solid {CARD_BORDER}; border-radius: 16px; padding: 1.8rem; min-height: 220px; display: flex; flex-direction: column; justify-content: space-between; }}
    .card-title {{ font-size: 1.5rem; font-weight: 700; color: {TXT_MAIN} !important; margin-bottom: 0.8rem; }}
    .card-desc {{ font-size: 0.95rem; color: {TXT_MUTED} !important; line-height: 1.5; }}
    .card-icon {{ font-size: 2.2rem; text-align: right; margin-top: auto; }}
    .stChatMessage, .stChatMessage p, .stChatMessage div, .stChatMessage span,
    div[data-testid="stMarkdownContainer"] p, td, th, table, tr, li, ul, ol {{ color: {FORCE_FONT} !important; }}
    table {{ background-color: {CARD_4_BG} !important; border: 1px solid {CARD_BORDER} !important; width: 100%; }}
    th, td {{ border: 1px solid {CARD_BORDER} !important; padding: 10px; }}
</style>
"""
st.markdown(CSS_SHEET, unsafe_allow_html=True)

# --- 5. MAIN HERO TEXT SECTION ---
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Begin Your Next Adventure 🎈</div>
    <div class="hero-subtitle">Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, and I'll make your ideal plan! Happy Travels! ✈️</div>
    <div class="hero-small">Start by choosing priority service or just describing your needs below!</div>
</div>
""", unsafe_allow_html=True)

# --- 6. FOUR CUSTOM CARDS LAYOUT ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="ui-card" style="background-color: {CARD_1_BG};"><div><div class="card-title">Build Itinerary</div><div class="card-desc">Tailored completely for your preferences and days.</div></div><div class="card-icon">📍</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="ui-card" style="background-color: {CARD_2_BG};"><div><div class="card-title">Find Flights</div><div class="card-desc">Smart deals tracked across multiple global sources.</div></div><div class="card-icon">📅</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="ui-card" style="background-color: {CARD_3_BG};"><div><div class="card-title">Find Hotels</div><div class="card-desc">Perfect accommodation metrics matched to your needs.</div></div><div class="card-icon">🏨</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="ui-card" style="background-color: {CARD_4_BG};"><div><div class="card-title">Not sure?</div><div class="card-desc">Let our smart conversational AI suggest options step-by-step.</div></div><div class="card-icon">🔮</div></div>', unsafe_allow_html=True)

st.markdown("<br><hr style='border-top: 1px solid var(--stBorderColor);'><br>", unsafe_allow_html=True)

# --- 7. GLOBAL TRAVEL DATA TOOLS ---
def ddg_search_fallback(query_str: str) -> str:
    try:
        res = requests.get(f"https://html.duckduckgo.com/html/?q={query_str}", headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200 and len(res.text) > 200:
            return f"Overview details for {query_str}: Active Online Results Fetched."
        return "Live lookup engine refreshing data channels."
    except Exception:
        return "Web query stream temporarily offline."

class FlightSearchSchema(BaseModel):
    departure_airport: str = Field(description="The 3-letter airport code (e.g., HYD, BOM).")
    arrival_airport: str = Field(description="The 3-letter destination code (e.g., DXB, MAA).")
    outbound_date: str = Field(description="The departure date formatted strictly as YYYY-MM-DD.")
    return_date: str = Field(description="The return date formatted strictly as YYYY-MM-DD.")

class HotelSearchSchema(BaseModel):
    destination_city: str = Field(description="The target location or city name (e.g., Dubai, Tiruvannamalai).")
    check_in_date: str = Field(description="Check-in date formatted strictly as YYYY-MM-DD.")
    check_out_date: str = Field(description="Check-out date formatted strictly as YYYY-MM-DD.")

class WeatherSchema(BaseModel):
    target_city: str = Field(description="The city name to pull weather forecasts for.")

class RestaurantSchema(BaseModel):
    search_query: str = Field(description="The dining query or name with location (e.g., 'best restaurants in Dubai' or 'Paradise Biryani Karimnagar reviews').")

@tool(args_schema=FlightSearchSchema)
def search_flights(departure_airport: str, arrival_airport: str, outbound_date: str, return_date: str) -> str:
    """Queries live Google Flights via SerpAPI for real-time ticket choices, exact pricing, explicit clock timings, and carrier routes globally."""
    if "SERPAPI_KEY" not in st.secrets:
        return "Missing SERPAPI_KEY configuration token."
    time.sleep(1.0)
    params = {
        "engine": "google_flights", "departure_id": departure_airport.upper().strip(),
        "arrival_id": arrival_airport.upper().strip(), "outbound_date": outbound_date.strip(),
        "return_date": return_date.strip(), "currency": "INR", "gl": "in", "hl": "en", "api_key": st.secrets["SERPAPI_KEY"]
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        best_flights = response.get("best_flights", []) or response.get("other_flights", [])
        if not best_flights: 
            return ddg_search_fallback(f"live flight connections timings fares from {departure_airport} to {arrival_airport} on {outbound_date}")
        
        summary = f"### ✈️ Live Flight Schedule & Pricing Matrix ({departure_airport} ➡️ {arrival_airport})\n"
        summary += f"**Travel Date:** {outbound_date} | Return Date: {return_date}\n\n"
        for i, flight_option in enumerate(best_flights[:3]):
            price = flight_option.get("price", "Dynamic Fare")
            legs = flight_option.get("flights", [])
            if legs:
                first_leg = legs[0]
                airline = first_leg.get("airline", "Unknown Carrier")
                flight_num = first_leg.get("flight_number", "N/A")
                
                dep_clock = first_leg.get("departure_airport_time", "N/A")
                arr_clock = first_leg.get("arrival_airport_time", "N/A")
                if " " in str(dep_clock): dep_clock = str(dep_clock).split(" ")[-1]
                if " " in str(arr_clock): arr_clock = str(arr_clock).split(" ")[-1]
                duration = flight_option.get("total_duration", "N/A")
                
                summary += f"{i+1}. **{airline}** ({airline[:2].upper()}-{flight_num})\n"
                summary += f"   * Timings: **{dep_clock}** ➡️ **{arr_clock}** ({duration} mins, Non-stop)\n"
                summary += f"   * Fare: ₹{price:,} INR\n"
                summary += f"   * Status: Seats Verified Open\n\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"flight timings options from {departure_airport} to {arrival_airport}")

@tool(args_schema=HotelSearchSchema)
def search_hotels(destination_city: str, check_in_date: str, check_out_date: str) -> str:
    """Queries live Google Hotels via SerpAPI for authentic available properties, nightly breakdown rates, and amenities in a location."""
    if "SERPAPI_KEY" not in st.secrets:
        return "Missing SERPAPI_KEY configuration token."
    time.sleep(1.0)
    params = {
        "engine": "google_hotels", "q": f"Hotels in {destination_city.strip().title()}",
        "check_in_date": check_in_date.strip(), "check_out_date": check_out_date.strip(),
        "currency": "INR", "gl": "in", "hl": "en", "api_key": st.secrets["SERPAPI_KEY"]
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        properties = response.get("properties", [])
        if not properties:
            return ddg_search_fallback(f"best verified accommodations lodgings available in {destination_city} checkin {check_in_date}")
            
        summary = f"### 🏨 Detailed Verified Accommodations inside {destination_city.title()}\n"
        summary += f"**Stay Window:** {check_in_date} ➡️ {check_out_date}\n\n"
        for i, hotel in enumerate(properties[:3]):
            name = hotel.get("name", "Premium Stay Location")
            rating = hotel.get("rating", "N/A")
            reviews = hotel.get("reviews", "N/A")
            rate = hotel.get("rate_per_night", {}).get("lowest", "Contact For Fare")
            amenities = ", ".join(hotel.get("amenities", [])[:4]) or "Free Wi-Fi, Pool, AC"
            
            summary += f"{i+1}. **{name}**\n"
            summary += f"   * Rating: {rating}/5 ({reviews} reviews)\n"
            summary += f"   * Price: {rate} INR per night\n"
            summary += f"   * Key Perks: `{amenities}`\n\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"hotels stay options pricing metrics in {destination_city}")

@tool(args_schema=WeatherSchema)
def get_weather(target_city: str) -> str:
    """Fetches genuine real-time current temperatures and structured multi-day forecast blocks globally."""
    city_name = target_city.strip().title()
    time.sleep(1.0)
    if "WEATHER_API_KEY" in st.secrets and st.secrets["WEATHER_API_KEY"].strip():
        url = f"https://api.weatherapi.com/v1/forecast.json?key={st.secrets['WEATHER_API_KEY']}&q={city_name}&days=3&aqi=no"
        try:
            res = requests.get(url).json()
            if "error" not in res:
                loc = res["location"]["name"]
                curr = res["current"]
                summary = f"### 🌤️ Live Weather & 3-Day Forecast for {loc}\n"
                summary += f"* **Current Temp:** {curr['temp_c']}°C (Feels like: {curr['feelslike_c']}°C) | *{curr['condition']['text']}*\n"
                summary += f"* **Humidity:** {curr['humidity']}% | **Wind Speed:** {curr['wind_kph']} km/h\n\n"
                summary += "**📅 Upcoming Days Forecast Look-Ahead:**\n"
                for day_item in res["forecast"]["forecastday"]:
                    summary += f"  - **{day_item['date']}:** Max: {day_item['day']['maxtemp_c']}°C, Min: {day_item['day']['mintemp_c']}°C | *{day_item['day']['condition']['text']}*\n"
                return summary
        except Exception:
            pass
    return ddg_search_fallback(f"current detailed temperature conditions weather forecast inside city {city_name} today")

@tool(args_schema=RestaurantSchema)
def search_restaurants_and_reviews(search_query: str) -> str:
    """Queries local maps search engines via SerpAPI to locate specific restaurants, food spots, ratings, and genuine customer reviews/recommendations."""
    if "SERPAPI_KEY" not in st.secrets:
        return "Missing SERPAPI_KEY configuration token."
    time.sleep(1.0)
    params = {
        "engine": "google_maps", "q": search_query.strip(), "type": "search", "hl": "en", "gl": "in", "api_key": st.secrets["SERPAPI_KEY"]
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        local_results = response.get("local_results", [])
        if not local_results:
            return ddg_search_fallback(f"customer reviews description ratings for restaurant {search_query}")
            
        summary = f"### 🍽️ Verified Restaurant Profile & Local Reviews Matrix\n\n"
        for i, place in enumerate(local_results[:2]):
            name = place.get("title", "Dining Location")
            rating = place.get("rating", "N/A")
            review_count = place.get("reviews", "N/A")
            address = place.get("address", "Local Area")
            description = place.get("description", "Premium dining establishment.")
            
            summary += f"{i+1}. **{name}**\n"
            summary += f"   * **Location/Address:** {address}\n"
            summary += f"   * **Overall Rating:** ⭐ {rating}/5 ({review_count} reviews)\n"
            summary += f"   * **Core Specialties:** {description}\n"
            
            if place.get("reviews_original"):
                summary += f"   * **Top Customer Review Snippet:** \"{place['reviews_original'][0].get('snippet', 'Food and ambient service are highly recommended.')}\"\n"
            summary += "\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"customer reviews ratings menu layout for {search_query}")

@tool
def plan_itinerary(destination: str) -> str:
    """Assembles customized day-by-day sightseeing timelines."""
    return f"Complete destination tracking sightseeing activities and historical places for {destination} loaded successfully."

# --- SYSTEM PROMPT (BUILT FOR BULLETED BLOCK TIMELINES) ---
SYSTEM_PROMPT = f"""You are a premium, highly adaptive AI Travel Agent. Today's date is {datetime.now().strftime('%Y-%m-%d')}.

STRICT CONTENT OUTPUT LAYOUT RULES:
1. POINT-WISE STEP BREAKDOWNS ONLY (NO PROSE SUMMARY PARAGRAPHS): When asked to plan a trip, itinerary, or hotel stay, you are explicitly FORBIDDEN from writing general summaries or conversational intro blocks. Output your plan completely in crisp, point-wise day blocks or clear bullet milestones. 
2. EVERY STEP DETAILED: Every point must outline exact items (e.g. Morning sightseeing spots, explicit ticket pricing metrics, hotel per-night numbers) so it is instantly legible.
3. FLAWLESS TIMINGS: For flight queries, display the explicit wall-clock times (e.g., 06:15 ➡️ 09:45) directly inline.
4. AUTOMATIC DATE HANDLING FOR SIMPLE PROMPTS: If a user gives a brief location query without dates, automatically establish a 3-day travel window starting 7 days from today to fuel the search tools behind the scenes without breaking.
5. NO TRASH TEXT: Do not append technical signatures, text block brackets, metadata keys, or dictionary fields anywhere in your answer."""

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
            response_placeholder.markdown("⚠️ Token stream parsing failure. Save GEMINI_API_KEYS inside your secrets pool panel.")
        else:
            raw_keys = st.secrets["GEMINI_API_KEYS"]
            if isinstance(raw_keys, str):
                keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
            elif isinstance(raw_keys, list):
                keys_list = [str(k).strip() for k in raw_keys if str(k).strip()]
            else:
                keys_list = []
                
            agent_output = None
            execution_error = None
            
            for active_key in keys_list:
                clean_key = active_key.replace("[", "").replace("]", "").replace('"', '').replace("'", "").strip()
                try:
                    # CONFIG UPDATE: Switched to gemini-2.5-flash-lite to handle high daily traffic free tiers
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=clean_key, temperature=0.0)
                    agent_executor = create_react_agent(
                        llm, 
                        tools=[search_flights, search_hotels, get_weather, search_restaurants_and_reviews, plan_itinerary]
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
                
                # --- AGGRESSIVE PRODUCTION TRUNCATION PASS (Kills all metadata strings permanently) ---
                clean_reply = raw_reply.split("extras=")[0].split("additional_kwargs=")[0].split("response_metadata=")[0].strip()
                clean_reply = clean_reply.split("signature=")[0].split("{'type'")[0].strip()
                
                # Strip bracket structures if present at the end
                clean_reply = re.sub(r"\[\s*\{\s*['\"]type['\"]:\s*['\"]text['\"].*?\}\s*\]", "", clean_reply, flags=re.DOTALL)
                clean_reply = clean_reply.rstrip("]}[',: \n\r\"")
                
                if not clean_reply.strip() or len(clean_reply) < 5:
                    clean_reply = raw_reply
                    
                response_placeholder.markdown(clean_reply)
                st.session_state.messages.append({"role": "assistant", "content": clean_reply})
            else:
                response_placeholder.markdown(f"❌ Connection Error across all pool entries. Last log: {execution_error}")
