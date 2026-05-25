import streamlit as st
import os
import re
import traceback

# --- CORE LOGIC: Importing backend components safely ---
from agent import get_agent, get_keys_pool

# --- 1. Page Configuration ---
st.set_page_config(page_title="Free AI Travel Agent", layout="wide", initial_sidebar_state="collapsed")

# Initialize Session Memory & Theme Slots Safely
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_destination" not in st.session_state:
    st.session_state.current_destination = ""
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "light"

# --- 2. Theme Toggle Controller Placement ---
toggle_col1, toggle_col2 = st.columns([8, 2])
with toggle_col2:
    theme_toggle = st.toggle("🌓 Dark Mode", value=(st.session_state.app_theme == "dark"))
    st.session_state.app_theme = "dark" if theme_toggle else "light"

# --- 3. Advanced Premium UI Style Selector ---
if st.session_state.app_theme == "dark":
    THEME_BG = "radial-gradient(circle at 15% 15%, #1e1b4b 0%, #311042 35%, #111827 100%)"
    TEXT_COLOR = "#f8fafc"
    CARD_BG = "#1f2937"
    CARD_BORDER = "#374151"
    SUB_TEXT_COLOR = "#94a3b8"
    CARD_YELLOW_BG = "#1f2937"
    CARD_BLUE_LT_BG = "#1f2937"
    CARD_BLUE_DK_BG = "#1f2937"
else:
    THEME_BG = "radial-gradient(circle at 15% 15%, #fee2e2 0%, #fae8ff 35%, #f5f3ff 65%, #e0f2fe 100%)"
    TEXT_COLOR = "#1e293b"
    CARD_BG = "#ffffff"
    CARD_BORDER = "#e2e8f0"
    SUB_TEXT_COLOR = "#64748b"
    CARD_YELLOW_BG = "linear-gradient(180deg, #fef08a 0%, #fefcd0 100%)"
    CARD_BLUE_LT_BG = "linear-gradient(180deg, #bfdbfe 0%, #eff6ff 100%)"
    CARD_BLUE_DK_BG = "linear-gradient(180deg, #93c5fd 0%, #dbeafe 100%)"

STYLE_SHEET = f"""
<style>
    .stApp {{ 
        background: {THEME_BG} !important; 
        color: {TEXT_COLOR}; 
        font-family: 'Inter', sans-serif; 
    }}
    @keyframes professionalGlideUp {{ 
        0% {{ opacity: 0; transform: translateY(20px); }} 
        100% {{ opacity: 1; transform: translateY(0); }} 
    }}
    .animated-element {{ animation: professionalGlideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both; }}
    .hero-container {{ text-align: center; padding-top: 1rem; padding-bottom: 1rem; }}
    .main-title {{ font-size: 2.6rem; font-weight: 800; color: #ea580c; margin-bottom: 0.5rem; letter-spacing: -0.5px; }}
    .sub-title {{ font-size: 1.1rem; color: {SUB_TEXT_COLOR}; font-weight: 500; max-width: 600px; margin: 0 auto 1.5rem auto; line-height: 1.6; }}
    div.stButton > button {{ background-color: transparent !important; border: none !important; padding: 0 !important; width: 100% !important; height: auto !important; text-align: left !important; box-shadow: none !important; }}
    div.stButton > button:hover {{ background-color: transparent !important; }}
    .feature-card {{ background-color: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 20px; padding: 2.2rem 1.6rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03); transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease; min-height: 230px; display: flex; flex-direction: column; justify-content: space-between; width: 100%; }}
    .feature-card:hover {{ transform: translateY(-8px) scale(1.03); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06); }}
    .card-yellow {{ background: {CARD_YELLOW_BG} !important; }}
    .card-blue-light {{ background: {CARD_BLUE_LT_BG} !important; }}
    .card-blue-dark {{ background: {CARD_BLUE_DK_BG} !important; }}
    .card-title {{ font-size: 1.6rem; font-weight: 700; color: {TEXT_COLOR}; margin-bottom: 0.6rem; }}
    .card-desc {{ font-size: 0.95rem; color: {SUB_TEXT_COLOR}; line-height: 1.5; }}
    .chat-container {{ max-width: 850px; margin: 2.5rem auto 6rem auto; padding: 1rem; }}
    .stChatMessage {{ background-color: {CARD_BG} !important; border: 1px solid {CARD_BORDER} !important; color: {TEXT_COLOR} !important; border-radius: 18px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.01) !important; margin-bottom: 1.2rem !important; padding: 1.2rem !important; animation: professionalGlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }}
    div[data-testid='stChatInput'] {{ position: fixed !important; bottom: 24px !important; left: 50% !important; transform: translateX(-50%) !important; width: 100% !important; max-width: 850px !important; z-index: 999999 !important; padding: 0 1rem !important; }}
    div[data-testid='stChatInput'] textarea {{ background-color: {CARD_BG} !important; color: {TEXT_COLOR} !important; border: 1px solid {CARD_BORDER} !important; border-radius: 30px !important; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.05) !important; padding: 14px 24px !important; transition: border-color 0.3s ease; }}
    div[data-testid='stChatInput'] textarea:focus {{ border-color: #ea580c !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 7rem !important; }}
</style>
"""
st.markdown(STYLE_SHEET, unsafe_allow_html=True)

HERO_LAYOUT = f"""
<div class='hero-container animated-element'>
    <div class='main-title'>Begin Your Next Adventure 🪂</div>
    <div class='sub-title'>Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, and I'll make your ideal plan! Happy Travels! ✈️</div>
</div>
"""
st.markdown(HERO_LAYOUT, unsafe_allow_html=True)

SUB_TEXT = f"<p class='animated-element' style='text-align:center; color:{SUB_TEXT_COLOR}; margin-top:-1rem; margin-bottom:2rem;'>Start by choosing priority service or just describing your needs below!</p>"
st.markdown(SUB_TEXT, unsafe_allow_html=True)

# --- 4. Math Processing Functions Node ---
def solve_complex_budget_allocation(liquidity_pool=500000, medical_transit=350000, voucher_rate=2500, supply_kit_rate=800, total_families=45, total_nights=2):
    total_voucher_cost = total_families * total_nights * voucher_rate
    initial_supply_cost = total_families * supply_kit_rate
    total_allocated_cost = medical_transit + total_voucher_cost + initial_supply_cost
    deficit = total_allocated_cost - liquidity_pool
    
    if deficit <= 0:
        return (
            f"### 📋 Emergency Evacuation Balanced Allocation Grid\n\n"
            f"* **Total Liquidity Pool:** ₹{liquidity_pool:,} INR\n"
            f"* **Total Projected Costs:** ₹{total_allocated_cost:,} INR\n"
            f"* **Net Pool Variance:** 🟢 Remaining Balance: ₹{abs(deficit):,} INR\n\n"
            f"| Asset Allocation Line | Unit Metrics Details | Calculated Line Subtotal |\n"
            f"| :--- | :--- | :--- |\n"
            f"| 🚑 Medical Transport | Flat Execution Fee | ₹{medical_transit:,} INR |\n"
            f"| 🏨 Hotel Vouchers | {total_families} Families × {total_nights} Nights @ ₹{voucher_rate}/nt | ₹{total_voucher_cost:,} INR |\n"
            f"| 📦 Dietary Supply Kits | {total_families} Kits allocated @ ₹{supply_kit_rate}/ea | ₹{initial_supply_cost:,} INR |"
        )
    
    available_for_kits = liquidity_pool - (medical_transit + total_voucher_cost)
    if available_for_kits < 0:
        return (
            f"### 🚨 CRITICAL DEFICIT: Unavoidable Budget Breach Matrix\n\n"
            f"**Total Liquidity Pool:** ₹{liquidity_pool:,} INR  \n"
            f"**Fixed Requirements (Transit + Stays):** ₹{medical_transit + total_voucher_cost:,} INR  \n"
            f"**Absolute Structural Deficit:** 🔴 ₹{abs(available_for_kits):,} INR\n\n"
            f"| Asset Allocation Line | Mitigated Metrics Details | Approved Line Subtotal |\n"
            f"| :--- | :--- | :--- |\n"
            f"| 🚑 Medical Transport | Fixed Core Essential Requirement | ₹{medical_transit:,} INR |\n"
            f"| 🏨 Hotel Vouchers | Fixed Housing Essential Requirement | ₹{total_voucher_cost:,} INR |\n"
            f"| 📦 Dietary Supply Kits | 🔴 Dropped Proportionally (0 Allocation) | ₹0 INR |"
        )

    reduced_kits = int(available_for_kits // supply_kit_rate)
    kits_subtotal = reduced_kits * supply_kit_rate
    actual_spent = medical_transit + total_voucher_cost + kits_subtotal
    remaining_pool_dust = liquidity_pool - actual_spent

    return (
        f"### ⚖️ Prioritized Cost-Cutting Mitigation Budget Grid\n\n"
        f"**Initial Calculated Def
