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
    #MainMenu
