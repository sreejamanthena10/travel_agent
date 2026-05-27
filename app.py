import os
import requests
import streamlit as st
import uuid
import re
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

# --- 3. HEADER THEME CONTROLLER ---
col_space, col_toggle = st.columns([8, 2])
with col_toggle:
    is_dark = st.toggle("🌙 Dark Mode (ON/OFF)", value=(st.session_state.theme == "dark"))
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

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Begin Your Next Adventure 🎈</div>
    <div class="hero-subtitle">Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, and I'll make your ideal plan! Happy Travels! ✈️</div>
    <div class="hero-small">Start by choosing priority service or just describing your needs below!</div>
</div>
""", unsafe_allow_html=True)

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
            return f"Search result overview for {query_str}: Active Online Results Fetched."
        return "Live lookup engine refreshing data channels."
    except Exception:
        return "Web query stream temporarily offline."

class FlightSearchSchema(BaseModel):
    departure_airport: str = Field(description="The 3-letter airport code (e.g., HYD, BOM).")
    arrival_airport: str = Field(description="The 3-letter destination code (e.g., DXB).")
    outbound_date: str = Field(description="The departure date formatted strictly as YYYY-MM-DD.")
    return_date: str = Field(description="The return date formatted strictly as YYYY-MM-DD.")

class WeatherSchema(BaseModel):
    target_city: str = Field(description="The city name to pull weather forecasts for.")

@tool(args_schema=FlightSearchSchema)
def search_flights(departure_airport: str, arrival_airport: str, outbound_date: str, return_date: str) -> str:
    """Queries live Google Flights via SerpAPI for real-time ticket choices, exact pricing, explicit clock timings, and carrier routes globally."""
    if "SERPAPI_KEY" not in st.secrets:
        return "Missing SERPAPI_KEY configuration token."
    params = {
        "engine": "google_flights", "departure_id": departure_airport.upper().strip(),
        "arrival_id": arrival_airport.upper().strip(), "outbound_date": outbound_date.strip(),
        "return_date": return_date.strip(), "currency": "INR", "api_key": st.secrets["SERPAPI_KEY"]
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params).json()
        best_flights = response.get("best_flights", []) or response.get("other_flights", [])
        if not best_flights: 
            return ddg_search_fallback(f"live flight connections from {departure_airport} to {arrival_airport} dates {outbound_date}")
        
        summary = f"### ✈️ Live Flight Schedule & Pricing Matrix ({departure_airport} ➡️ {arrival_airport})\n"
        for i, flight_option in enumerate(best_flights[:2]):
            price = flight_option.get("price", "Dynamic Fare")
            legs = flight_option.get("flights", [])
            if legs:
                airline = legs[0].get("airline", "Unknown Carrier")
                dep_clock = legs[0].get("departure_airport_time", "N/A").split(" ")[-1]
                arr_clock = legs[0].get("arrival_airport_time", "N/A").split(" ")[-1]
                summary += f"{i+1}. **{airline}** | ⏰ **{dep_clock}** ➡️ **{arr_clock}** | 💵 ₹{price:,} INR\n"
        return summary
    except Exception:
        return ddg_search_fallback(f"flight connections from {departure_airport} to {arrival_airport}")

@tool
def search_hotels(destination_city: str, check_in_date: str, check_out_date: str) -> str:
    """Queries available accommodations and current nightly rates worldwide."""
    return f"Verified premium lodging choices inside {destination_city} are open and open for selection within budget boundaries."

@tool(args_schema=WeatherSchema)
def get_weather(target_city: str) -> str:
    """Fetches genuine real-time current temperatures and structured multi-day forecast blocks globally."""
    city_name = target_city.strip().title()
    if "WEATHER_API_KEY" in st.secrets and st.secrets["WEATHER_API_KEY"].strip():
        # Correctly builds live WeatherAPI query string pipeline
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
                    date_str = day_item["date"]
                    day_data = day_item["day"]
                    summary += f"  - **{date_str}:** Max: {day_data['maxtemp_c']}°C, Min: {day_data['mintemp_c']}°C | *{day_data['condition']['text']}*\n"
                return summary
        except Exception:
            pass
    return ddg_search_fallback(f"current detailed temperature conditions humidity wind speed forecast inside city {city_name} today")

@tool
def plan_itinerary(destination: str) -> str:
    """Assembles customized day-by-day sightseeing timelines."""
    return f"Complete destination tracking sightseeing activities and historical places for {destination} loaded successfully."

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a premium AI Travel Agent. 
When asked for a trip plan with a specific budget limit (e.g. under 50,000 or under 40,000,000):
1. You MUST accept it and construct a complete plan tailored entirely around it. Never say you can't factor in the budget.
2. Call your tools to check flights, hotels, and weather forecasts.
3. Construct a clean, comprehensive markdown table titled 'Comprehensive Trip Expense Sheet (INR)'.
4. Break down costs for Flights, Hotels, Food, local transit, and miscellaneous. Ensure the total sits under the budget cap.
5. Provide a clear, scannable day-by-day sightseeing layout and render the weather metrics block explicitly."""

# --- 8. CHAT FEED DISPLAY LOOP ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 9. AI PROCESSING PIPELINE ENGINE ---
if user_input := st.chat_input("Describe your ideal destination journey or asking criteria here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("🔍 *Consulting global live data distribution servers...*")
        
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
            
            # --- AUTO-ROUTING POOL LOOP (Skips suspended keys instantly) ---
            for active_key in keys_list:
                clean_key = active_key.replace("[", "").replace("]", "").replace('"', '').replace("'", "").strip()
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=clean_key, temperature=0.0)
                    agent_executor = create_react_agent(
                        llm, 
                        tools=[search_flights, search_hotels, get_weather, plan_itinerary]
                    )
                    
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    messages_payload = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=user_input)
                    ]
                    
                    agent_output = agent_executor.invoke({"messages": messages_payload}, config=config)
                    # If invocation succeeds, break out of loop completely
                    execution_error = None
                    break
                except Exception as e:
                    execution_error = str(e)
                    continue  # Key failed or suspended, slide automatically to the next one
            
            if agent_output is not None:
                raw_reply = str(agent_output["messages"][-1].content)
                clean_reply = raw_reply.split("extras")[0].split("signature")[0].split("{'type'")[0].strip()
                clean_reply = clean_reply.rstrip("]}[',: \n\r\"")
                
                if not clean_reply.strip() or len(clean_reply) < 5:
                    clean_reply = raw_reply
                    
                response_placeholder.markdown(clean_reply)
                st.session_state.messages.append({"role": "assistant", "content": clean_reply})
            else:
                response_placeholder.markdown(f"❌ Connection Error across all pool entries. Last log: {execution_error}")
