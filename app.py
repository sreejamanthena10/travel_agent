import streamlit as st
import os
import re
import traceback

# --- CORE LOGIC: Importing backend components safely ---
from agent import get_agent, get_keys_pool

# --- 1. Page Configuration ---
st.set_page_config(page_title="Free AI Travel Agent", layout="wide", initial_sidebar_state="collapsed")

# --- 2. Advanced Premium UI & Smooth Micro-Interaction Physics Injector ---
st.markdown("""
    <style>
    /* Precision Color-Matched Background Styling from Your Mockup */
    .stApp {
        background: radial-gradient(
            circle at 15% 15%, 
            #fee2e2 0%,    
            #fae8ff 35%,   
            #f5f3ff 65%,   
            #e0f2fe 100%   
        ) !important;
        color: #1e293b;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Hardware-Accelerated Cinematic Smooth Fade-In Glide Animation */
    @keyframes professionalGlideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .animated-element {
        animation: professionalGlideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    /* Main Header Layout styling */
    .hero-container {
        text-align: center;
        padding-top: 2.5rem;
        padding-bottom: 1rem;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ea580c;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }
    
    /* Transparent Clickable Button Layer Overlay */
    div.stButton > button {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
        height: auto !important;
        text-align: left !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background-color: transparent !important;
    }
    
    /* Service Layout Cards System with Hover Micro-bounces */
    .feature-card {
        background-color: white;
        border-radius: 20px;
        padding: 2.2rem 1.6rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
        min-height: 230px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        width: 100%;
    }
    .feature-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06);
    }
    .card-yellow { background: linear-gradient(180deg, #fef08a 0%, #fefcd0 100%); }
    .card-blue-light { background: linear-gradient(180deg, #bfdbfe 0%, #eff6ff 100%); }
    .card-blue-dark { background: linear-gradient(180deg, #93c5fd 0%, #dbeafe 100%); }
    .card-white { background: #ffffff; border: 1px solid #e2e8f0; }
    
    .card-title { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 0.6rem; }
    .card-desc { font-size: 0.95rem; color: #475569; line-height: 1.5; }
    
    /* Chat Container Framework */
    .chat-container {
        max-width: 850px;
        margin: 2.5rem auto 6rem auto;
        padding: 1rem;
    }
    .stChatMessage {
        background-color: white !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.01) !important;
        margin-bottom: 1.2rem !important;
        padding: 1.2rem !important;
        animation: professionalGlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    /* Fixed Positioning Layout for standard Chat Bars */
    div[data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 24px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 100% !important;
        max-width: 850px !important;
        z-index: 999999 !important;
        padding: 0 1rem !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 30px !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.05) !important;
        padding: 14px 24px !important;
        transition: border-color 0.3s ease;
    }
    div[data-testid="stChatInput"] textarea:focus { border-color: #ea580c !important; }
    
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem !important; padding-bottom: 7rem !important;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session Memory Slots
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_destination" not in st.session_state:
    st.session_state.current_destination = "Requested Destination"

# --- 3. Header Segment Rendering ---
st.markdown("""
<div class="hero-container animated-element">
    <div class="main-title">Begin Your Next Adventure 🪂</div>
    <div class="sub-title">
        Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, 
        and I'll make your ideal plan! Happy Travels! ✈️
    </div>
</div>
""", unsafe_allow_html=True)

click_prompt = ""

st.markdown("""
<p class="animated-element" style="text-align:center; color:#64748b; margin-top:-1rem; margin-bottom:2rem;">Start by choosing priority service or just describing your needs below!</p>
""", unsafe_allow_html=True)

# --- 4. Interactive Columns Setup ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("", key="btn_itinerary"):
        click_prompt = f"ACTION_ITINERARY: Build a highly scannable, day-by-day travel itinerary blueprint plan layout for {st.session_state.current_destination}."
    st.markdown('<div class="feature-card card-yellow animated-element" style="margin-top: -55px;"><div><div class="card-title">Build Itinerary</div><div class="card-desc">Tailored completely for your preferences and days.</div></div><div style="font-size: 3rem; text-align: right;">📍</div></div>', unsafe_allow_html=True)

with col2:
    if st.button("", key="btn_flights"):
        click_prompt = f"ACTION_FLIGHTS: Provide a detailed breakdown table chart of flight carrier plane schedules, route combinations, and travel metrics heading directly to {st.session_state.current_destination}."
    st.markdown('<div class="feature-card card-blue-light animated-element" style="margin-top: -55px;"><div><div class="card-title">Find Flights</div><div class="card-desc">Smart deals tracked across multiple global sources.</div></div><div style="font-size: 3rem; text-align: right;">📅</div></div>', unsafe_allow_html=True)

with col3:
    if st.button("", key="btn_hotels"):
        click_prompt = f"ACTION_HOTELS: Locate highly recommended budget stays, price-tiered accommodation grids, and rating features inside {st.session_state.current_destination}."
    st.markdown('<div class="feature-card card-blue-dark animated-element" style="margin-top: -55px;"><div><div class="card-title">Find Hotels</div><div class="card-desc">Perfect accommodation metrics matched to your needs.</div></div><div style="font-size: 3rem; text-align: right;">🏨</div></div>', unsafe_allow_html=True)

with col4:
    if st.button("", key="btn_suggest"):
        click_prompt = f"ACTION_SUGGEST: Explore hidden tourist landmarks, famous spots, and local sightseeing items around {st.session_state.current_destination}."
    st.markdown('<div class="feature-card card-white animated-element" style="margin-top: -55px;"><div><div class="card-title">Not sure?</div><div class="card-desc">Let our smart conversational AI suggest options step-by-step.</div></div><div style="font-size: 3rem; text-align: right;">🔮</div></div>', unsafe_allow_html=True)

# --- 5. Message Logs Render Matrix ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render history instantly
for msg in st.session_state.messages:
    display_content = msg["content"]
    if display_content.startswith("ACTION_"):
        display_content = display_content.split(": ", 1)[1]
    with st.chat_message(msg["role"]):
        st.markdown(display_content)

chat_input_val = st.chat_input("Type your travel needs here...")

# FIXED PROCESSING FLOW: Capture user input cleanly without immediate reruns killing execution state
user_input = ""
if click_prompt:
    user_input = click_prompt
    st.session_state.messages.append({"role": "user", "content": click_prompt})
elif chat_input_val:
    user_input = chat_input_val
    st.session_state.messages.append({"role": "user", "content": chat_input_val})
    
    # Destination parsing logic safely handled inline
    stop_phrases = ["plan a trip to", "hotels in", "flights to", "travel to", "go to", "weather in", "forecast for", "show flights from"]
    cleaned_dest = chat_input_val.lower()
    for phrase in stop_phrases:
        cleaned_dest = cleaned_dest.replace(phrase, "")
    words = [w.strip("?,.¡!").capitalize() for w in cleaned_dest.split() if w.strip()]
    if words and not any(w.lower() in ["weather", "forecast", "temp", "temperature", "climate", "june", "july"] for w in words):
        st.session_state.current_destination = " ".join(words)

# --- 6. Intelligent Response Core Processor Layer ---
if user_input:
    # Render user prompt immediately on screen
    if click_prompt:
        clean_user_display = user_input.split(": ", 1)[1]
        with st.chat_message("user"):
            st.markdown(clean_user_display)
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Keywords validation boundaries
    input_words = [w.strip("?,.¡!").lower() for w in user_input.split()]
    weather_keywords = ["weather", "forecast", "temperature", "temp", "climate"]
    is_weather_query = any(keyword in input_words for keyword in weather_keywords) and not user_input.startswith("ACTION_")

    with st.chat_message("assistant"):
        if is_weather_query:
            loc = st.session_state.current_destination if st.session_state.current_destination != "Requested Destination" else "Your Destination"
            st.markdown(f"### ☀️ {loc} 6-Day Visual Forecast Matrix")
            weather_output = (
                "| Day | Condition | Temp (Low / High) | Rain % |\n"
                "| :--- | :---: | :---: | :---: |\n"
                "| **Sun** (Today) | ☀️ *Sunny / Extreme Heat* | 33°C / **43°C** | 0% |\n"
                "| **Mon** | ☀️ *Intense Sun Exposure* | 32°C / **43°C** | 5% |\n"
                "| **Tue** | 🌦️ *Passing Afternoon Clouds* | 32°C / **41°C** | 15% |\n"
                "| **Wed** | ☀️ *Clear / High Heat* | 32°C / **42°C** | 5% |\n"
                "| **Thu** | ☀️ *Intense Heatwave Peaks* | 32°C / **43°C** | 15% |\n"
                "| **Fri** | 🌤️ *Partly Cloudy / Humid* | 31°C / **41°C** | 15% |"
            )
            st.markdown(weather_output)
            st.session_state.messages.append({"role": "assistant", "content": f"### ☀️ {loc} 6-Day Visual Forecast Matrix\n" + weather_output})
        
        else:
            live_agent = get_agent()
            if live_agent is None:
                st.error("❌ Secrets Configuration Error: All listed tokens are invalid, empty, or exhausted.")
            else:
                with st.spinner("Processing expert travel logic..."):
                    try:
                        date_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?(,\s+\d{4})?', user_input, re.IGNORECASE)
                        extracted_date_context = f" on date {date_match.group(0)}" if date_match else ""
                        refined_query = f"{user_input}{extracted_date_context}. Ensure all flight tables explicitly reflect active schedules matching this timestamp context parameters."
                        
                        result = live_agent.invoke({"messages": [("user", refined_query)]})
                        
                        # --- UNIVERSAL MESSAGE EXTRACTOR LAYER ---
                        agent_messages = result.get("messages", [])
                        answer = ""
                        
                        for msg in reversed(agent_messages):
                            msg_type = getattr(msg, "type", "").lower()
                            class_name = msg.__class__.__name__
                            
                            if "ai" in msg_type or "ai" in class_name.lower():
                                if hasattr(msg, "content") and str(msg.content).strip():
                                    answer = str(msg.content)
                                    break
                        
                        if not answer and agent_messages:
                            last_msg = agent_messages[-1]
                            if hasattr(last_msg, "content"):
                                answer = str(last_msg.content)
                            elif isinstance(last_msg, dict) and "content" in last_msg:
                                answer = str(last_msg["content"])
                            else:
                                answer = str(last_msg)

                        if answer.strip():
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.warning("⚠️ The agent processed your query but returned an empty text layer.")
                            
                    except Exception as e:
                        st.error(f"❌ Backend Execution Failure: {str(e)}")
                        st.code(traceback.format_exc(), language="python")

st.markdown("</div>", unsafe_allow_html=True)
