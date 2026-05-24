import streamlit as st
import os

# --- CORE LOGIC: Importing your perfectly working backend components ---
from agent import get_agent, get_keys_pool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration ---
st.set_page_config(page_title="Free AI Travel Agent", layout="wide", initial_sidebar_state="collapsed")

# --- 2. Premium UI Design & Layout Injector ---
st.markdown("""
    <style>
    /* Global App Background Styling */
    .stApp {
        background: linear-gradient(135deg, #fce7f3 0%, #fae8ff 50%, #e0f2fe 100%);
        color: #1e293b;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Main Header Layout */
    .hero-container {
        text-align: center;
        padding-top: 2.5rem;
        padding-bottom: 1rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ea580c;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.5;
    }
    
    /* Transparent Clickable Button Wrapping Over CSS Cards */
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
        border: none !important;
    }
    
    /* Service Layout Cards System */
    .feature-card {
        background-color: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        width: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08);
    }
    .card-yellow { background: linear-gradient(180deg, #fef08a 0%, #fefcd0 100%); }
    .card-blue-light { background: linear-gradient(180deg, #bfdbfe 0%, #eff6ff 100%); }
    .card-blue-dark { background: linear-gradient(180deg, #93c5fd 0%, #dbeafe 100%); }
    .card-white { background: #ffffff; border: 1px solid #f1f5f9; }
    
    .card-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.4;
    }
    
    /* Chat Message Interface Formatting */
    .chat-container {
        max-width: 850px;
        margin: 0 auto 5rem auto;
        padding: 1rem;
    }
    .stChatMessage {
        background-color: white !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
        margin-bottom: 1rem !important;
        padding: 1rem !important;
    }
    
    /* State Prompt Alert Box Styling */
    .state-prompt {
        background-color: rgba(255, 255, 255, 0.85);
        border-left: 5px solid #ea580c;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Reposition Floating Input Bar to Screen Bottom */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 850px;
        z-index: 99;
        padding: 0 1rem;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 30px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06) !important;
        padding: 12px 20px !important;
    }
    
    /* Clean up default Streamlit branding layout elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem !important; padding-bottom: 6rem !important;}
    </style>
""", unsafe_allow_html=True)

# Initialize interactive state parameters inside memory storage
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "General"  # Options: "General", "Hotels", "Flights", "Itinerary"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Render Top Branding Hero Content ---
st.markdown("""
<div class="hero-container">
    <div class="main-title">Begin Your Next Adventure 🪂</div>
    <div class="sub-title">
        Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, 
        and I'll make your ideal plan! Happy Travels! ✈️<br>
        <span style="font-size: 0.9rem; color: #64748b;">Start by choosing priority service or just describing your needs below!</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4. Render Service Display Cards via Clickable Columns System ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    card1 = st.button("", key="btn_itinerary")
    st.markdown("""
    <div class="feature-card card-yellow" style="margin-top: -55px;">
        <div>
            <div class="card-title">Build Itinerary</div>
            <div class="card-desc">Tailored completely for your preferences and days.</div>
        </div>
        <div style="font-size: 3rem; text-align: right;">📍</div>
    </div>
    """, unsafe_allow_html=True)
    if card1:
        st.session_state.active_mode = "Itinerary"
        st.session_state.messages.append({"role": "assistant", "content": "🔮 **Itinerary Mode Activated:** Where would you like to plan your journey, and for how many days?"})

with col2:
    card2 = st.button("", key="btn_flights")
    st.markdown("""
    <div class="feature-card card-blue-light" style="margin-top: -55px;">
        <div>
            <div class="card-title">Find Flights</div>
            <div class="card-desc">Smart deals tracked across multiple global sources.</div>
        </div>
        <div style="font-size: 3rem; text-align: right;">📅</div>
    </div>
    """, unsafe_allow_html=True)
    if card2:
        st.session_state.active_mode = "Flights"
        st.session_state.messages.append({"role": "assistant", "content": "✈️ **Flight Search Mode Activated:** Please specify your departure city, destination, and ideal travel dates/budget."})

with col3:
    card3 = st.button("", key="btn_hotels")
    st.markdown("""
    <div class="feature-card card-blue-dark" style="margin-top: -55px;">
        <div>
            <div class="card-title">Find Hotels</div>
            <div class="card-desc">Perfect accommodation metrics matched to your needs.</div>
        </div>
        <div style="font-size: 3rem; text-align: right;">🏨</div>
    </div>
    """, unsafe_allow_html=True)
    if card3:
        st.session_state.active_mode = "Hotels"
        st.session_state.messages.append({"role": "assistant", "content": "🏨 **Hotel Matrix Mode Activated:** Which location are you traveling to, and what is your target budget per night?"})

with col4:
    card4 = st.button("", key="btn_suggest")
    st.markdown("""
