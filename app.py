import streamlit as st
import os
import re
import traceback

# --- CORE LOGIC: Importing backend components safely ---
from agent import get_agent, get_keys_pool

# --- 1. Page Configuration ---
st.set_page_config(page_title="Free AI Travel Agent", layout="wide", initial_sidebar_state="collapsed")

# --- 2. Advanced Premium UI & Smooth Micro-Interaction Physics Injector ---
st.markdown("<style>.stApp { background: radial-gradient(circle at 15% 15%, #fee2e2 0%, #fae8ff 35%, #f5f3ff 65%, #e0f2fe 100%) !important; color: #1e293b; font-family: 'Inter', sans-serif; } @keyframes professionalGlideUp { 0% { opacity: 0; transform: translateY(20px); } 100% { opacity: 1; transform: translateY(0); } } .animated-element { animation: professionalGlideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both; } .hero-container { text-align: center; padding-top: 2.5rem; padding-bottom: 1rem; } .main-title { font-size: 2.6rem; font-weight: 800; color: #ea580c; margin-bottom: 0.5rem; letter-spacing: -0.5px; } .sub-title { font-size: 1.1rem; color: #475569; font-weight: 500; max-width: 600px; margin: 0 auto 1.5rem auto; line-height: 1.6; } div.stButton > button { background-color: transparent !important; border: none !important; padding: 0 !important; width: 100% !important; height: auto !important; text-align: left !important; box-shadow: none !important; } div.stButton > button:hover { background-color: transparent !important; } .feature-card { background-color: white; border-radius: 20px; padding: 2.2rem 1.6rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03); transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease; min-height: 230px; display: flex; flex-direction: column; justify-content: space-between; width: 100%; } .feature-card:hover { transform: translateY(-8px) scale(1.03); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06); } .card-yellow { background: linear-gradient(180deg, #fef08a 0%, #fefcd0 100%); } .card-blue-light { background: linear-gradient(180deg, #bfdbfe 0%, #eff6ff 100%); } .card-blue-dark { background: linear-gradient(180deg, #93c5fd 0%, #dbeafe 100%); } .card-white { background: #ffffff; border: 1px solid #e2e8f0; } .card-title { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 0.6rem; } .card-desc { font-size: 0.95rem; color: #475569; line-height: 1.5; } .chat-container { max-width: 850px; margin: 2.5rem auto 6rem auto; padding: 1rem; } .stChatMessage { background-color: white !important; border-radius: 18px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.01) !important; margin-bottom: 1.2rem !important; padding: 1.2rem !important; animation: professionalGlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; } div[data-testid='stChatInput'] { position: fixed !important; bottom: 24px !important; left: 50% !important; transform: translateX(-50%) !important; width: 100% !important; max-width: 850px !important; z-index: 999999 !important; padding: 0 1rem !important; } div[data-testid='stChatInput'] textarea { background-color: white !important; color: #1e293b !important; border: 1px solid #cbd5e1 !important; border-radius: 30px !important; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.05) !important; padding: 14px 24px !important; transition: border-color 0.3s ease; } div[data-testid='stChatInput'] textarea:focus { border-color: #ea580c !important; } #MainMenu, footer, header {visibility: hidden;} .block-container {padding-top: 1rem !important; padding-bottom: 7rem !important;}</style>", unsafe_allow_html=True)

# Initialize Session Memory Slots
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_destination" not in st.session_state:
    st.session_state.current_destination = "Requested Destination"

# --- 3. Header Segment Rendering ---
st.markdown("<div class='hero-container
