import streamlit as st
import os

# --- FROM WEEK 2/3: Import your LangGraph multi-tool agent brain ---
from agent import get_agent 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration & Premium Executive CSS Styling ---
st.set_page_config(page_title="AeroConcierge AI", layout="centered")

# Advanced Glassmorphic Dark-Mode UI Theme Injection
st.markdown("""
    <style>
    /* Global Background Canvas */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Smooth Cubic Focus Animation */
    @keyframes ultraFadeIn {
        0% { opacity: 0; filter: blur(6px); transform: translateY(12px); }
        100% { opacity: 1; filter: blur(0px); transform: translateY(0); }
    }
    .animated-container {
        animation: ultraFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* Professional Corporate Header Layout */
    .main-header {
        font-size: 2.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        background: linear-gradient(135deg, #ffffff 20%, #60a5fa 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 500;
    }
    .header-line {
        height: 2px;
        width: 50px;
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        margin: 0 auto 1.2rem auto;
        border-radius: 10px;
    }
    
    /* Chat Input Field UI Overrides */
    input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 10px rgba(129, 140, 248, 0.3) !important;
    }
    
    /* Modernized Chat Message Bubble Style Tweaks */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        padding: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Render Animated Header Interface ---
st.markdown('<div class="animated-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">AeroConcierge AI</h1>', unsafe_allow_html=True)
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous Travel Intelligence & Verified Vector RAG Platform</p>', unsafe_allow_html=True)

# --- 3. Secure Production Key Injection ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("⚠️ Environment Configuration Missing: Please set 'GEMINI_API_KEY' in your Streamlit Advanced Settings.")
    st.stop()

# --- 4. Caching Layers for Maximum Performance Optimization ---
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
            loader = PyPDFLoader(file_path)
            all_pages.extend(loader.load_and_split())
            
    if all_pages:
        # PRODUCTION MIGRATION FIX: Synced embedding name core
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = FAISS.from_documents([all_pages[0]], embeddings)
        if len(all_pages) > 1:
            for page in all_pages[1:]:
                vector_db.add_documents([page])
        return vector_db
    return None

@st.cache_resource
def get_cached_agent(_key):
    os.environ["GOOGLE_API_KEY"] = _key
    return get_agent()

# PERFORMANCE FIX: Cache search operations to prevent UI loop lag
@st.cache_data
def fast_vector_search(_query, _key):
    os.environ["GOOGLE_API_KEY"] = _key
    vector_db = load_data(_key)
    if vector_db:
        docs = vector_db.similarity_search(_query, k=2) 
        return "\n".join([d.page_content for d in docs])
    return ""

if "messages" not in st.session_state
