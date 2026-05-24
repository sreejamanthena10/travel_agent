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
    /* Precision Color-Matched Background Styling from Screenshot (93) */
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
        margin: 2rem auto 5rem auto;
        padding: 1rem;
    }
    .stChatMessage {
        background-color: white !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
        margin-bottom: 1rem !important;
        padding: 1rem !important;
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

# Initialize persistent destination trackers in memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_destination" not in st.session_state:
    st.session_state.current_destination = "Africa" # Smart default context setting

# --- 3. Render Top Branding Hero Content ---
st.markdown("""
<div class="hero-container">
    <div class="main-title">Begin Your Next Adventure 🪂</div>
    <div class="sub-title">
        Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, 
        and I'll make your ideal plan! Happy Travels! ✈️
    </div>
</div>
""", unsafe_allow_html=True)

# Latch variable for monitoring button selections
click_prompt = ""

# --- 4. Render Service Display Cards System ---
st.markdown(f'<div style="text-align:center; color:#ea580c; font-weight:600; font-size:1.05rem; margin-top:-1rem; margin-bottom:2rem;">📍 Active Target Tracked: {st.session_state.current_destination}</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("", key="btn_itinerary"):
        click_prompt = f"Build a comprehensive scannability-optimized daily travel itinerary layout for: {st.session_state.current_destination}"
        st.session_state.messages.append({"role": "user", "content": f"📍 Build full Itinerary for **{st.session_state.current_destination}**"})
    st.markdown('<div class="feature-card card-yellow" style="margin-top: -55px;"><div><div class="card-title">Build Itinerary</div><div class="card-desc">Tailored completely for your preferences and days.</div></div><div style="font-size: 3rem; text-align: right;">📍</div></div>', unsafe_allow_html=True)

with col2:
    if st.button("", key="btn_flights"):
        click_prompt = f"Find flight travel route options, tracking deals, carrier connections, and pricing structures for: {st.session_state.current_destination}"
        st.session_state.messages.append({"role": "user", "content": f"📅 Search Flights to **{st.session_state.current_destination}**"})
    st.markdown('<div class="feature-card card-blue-light" style="margin-top: -55px;"><div><div class="card-title">Find Flights</div><div class="card-desc">Smart deals tracked across multiple global sources.</div></div><div style="font-size: 3rem; text-align: right;">📅</div></div>', unsafe_allow_html=True)

with col3:
    if st.button("", key="btn_hotels"):
        click_prompt = f"Find a detailed budget hotel matrix with multiple comfort choices, nightly rates, and features for destination: {st.session_state.current_destination}"
        st.session_state.messages.append({"role": "user", "content": f"🏨 Find Budget Hotels in **{st.session_state.current_destination}**"})
    st.markdown('<div class="feature-card card-blue-dark" style="margin-top: -55px;"><div><div class="card-title">Find Hotels</div><div class="card-desc">Perfect accommodation metrics matched to your needs.</div></div><div style="font-size: 3rem; text-align: right;">🏨</div></div>', unsafe_allow_html=True)

with col4:
    if st.button("", key="btn_suggest"):
        click_prompt = f"Show me top landmarks, unique highlights, and sightseeing items near: {st.session_state.current_destination}"
        st.session_state.messages.append({"role": "user", "content": f"🔮 Explore Sights near **{st.session_state.current_destination}**"})
    st.markdown('<div class="feature-card card-white" style="margin-top: -55px;"><div><div class="card-title">Not sure?</div><div class="card-desc">Let our smart conversational AI suggest options step-by-step.</div></div><div style="font-size: 3rem; text-align: right;">🔮</div></div>', unsafe_allow_html=True)

# --- 5. AGENT CONFIGURATION UTILITIES ---
keys_list = get_keys_pool()

@st.cache_resource
def load_data(_key): 
    os.environ["GOOGLE_API_KEY"] = _key
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "data", "raw")
    all_pages = []
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        for f in files:
            file_path = os.path.join(data_folder, f)
            try:
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load_and_split())
            except Exception:
                continue
    if all_pages:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return FAISS.from_documents([all_pages[0]], embeddings)
    return None

try:
    if "agent" not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = get_agent()
except Exception:
    st.session_state.agent = None

# Wrap chat viewport
