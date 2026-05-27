import streamlit as st
from agent import get_agent

# --- 1. System Layout Configurations ---
st.set_page_config(page_title="Concierge AI Travel Agent", page_icon="✈️", layout="wide")

# --- 2. Initialize App State Layouts ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your professional AI Travel Concierge. How can I assist you with your trip itineraries, real-time hotel lookups, or live flight configurations today?"}]

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# --- 3. Header Controls & Manual Light/Dark Theme Switcher ---
col_title, col_toggle = st.columns([8, 2])

with col_toggle:
    current_theme = st.session_state.theme
    toggle_text = "🌙 Night Sky Mode" if current_theme == "light" else "☀️ Bright Day Mode"
    if st.button(toggle_text, use_container_width=True):
        st.session_state.theme = "dark" if current_theme == "light" else "light"
        st.rerun()

# --- 4. System-Theme-Independent Custom CSS Injection ---
if st.session_state.theme == "dark":
    BG_STYLE = "radial-gradient(circle at 15% 15%, #1e1b4b 0%, #2e1042 40%, #0f172a 100%)"
    TXT_MAIN = "#ffffff"
    TXT_MUTED = "#94a3b8"
    BOX_BG = "#1e293b"
    BOX_BORDER = "#334155"
    FORCE_FONT = "#ffffff"
else:
    BG_STYLE = "radial-gradient(circle at 15% 15%, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%)"
    TXT_MAIN = "#0f172a"
    TXT_MUTED = "#475569"
    BOX_BG = "#ffffff"
    BOX_BORDER = "#cbd5e1"
    FORCE_FONT = "#0f172a"

CSS_SHEET = f"""
<style>
    /* Absolute global theme resets */
    .stApp {{
        background: {BG_STYLE} !important;
        color: {TXT_MAIN} !important;
    }}
    
    /* Text styling rules */
    h1, h2, h3, p, span, li, label, div {{
        color: {TXT_MAIN};
    }}
    
    /* Grid layout info cards */
    .feature-card {{
        background-color: {BOX_BG} !important;
        border: 1px solid {BOX_BORDER} !important;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }}
    
    /* Crucial visibility override: forces explicit text color within chat windows and markdown nodes */
    .stChatMessage, .stChatMessage p, .stChatMessage div, .stChatMessage span,
    div[data-testid="stMarkdownContainer"] p, td, th, table, tr, li, ul, ol {{
        color: {FORCE_FONT} !important;
    }}
    
    table {{
        background-color: {BOX_BG} !important;
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
    }}
    th, td {{
        border: 1px solid {BOX_BORDER} !important;
        padding: 8px 12px;
    }}
</style>
"""
st.markdown(CSS_SHEET, unsafe_allow_html=True)

# --- 5. Main Hero Display Banner ---
with col_title:
    st.markdown("<h1 style='margin:0; font-weight:800; color:#f97316;'>✈️ GLOBAL TRAVEL CONCIERGE AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='margin:0 0 1.5rem 0; font-size:1.1rem; color:{TXT_MUTED};'>Professional Agent Suite — Powered by Live Google Flights, Hotels, and Weather API Channels.</p>", unsafe_allow_html=True)

# --- 6. Quick Action Suggestion Cards ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='feature-card'><b style='color:#f97316;'>🗺️ Smart Destination Planning</b><br><span style='font-size:0.9rem; color:{TXT_MUTED};'>Ask: 'Plan a budget-friendly spiritual trip to Arunachalam and show nearby historical places.'</span></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='feature-card'><b style='color:#3b82f6;'>🏨 Real-Time Lodging Rates</b><br><span style='font-size:0.9rem; color:{TXT_MUTED};'>Ask: 'Show verified available hotels in Singapore with rating summaries.'</span></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='feature-card'><b style='color:#10b981;'>🌤️ Telemetry & Live Flights</b><br><span style='font-size:0.9rem; color:{TXT_MUTED};'>Ask: 'Check current flight price matrix from HYD and local climate weather updates.'</span></div>", unsafe_allow_html=True)

# --- 7. Chat Feed Rendering ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 8. AI Agent Execution Pipeline ---
if user_input := st.chat_input("Ask about flight routes, hotel options, weather updates, or tailored itineraries..."):
    # Render user query immediately to meet the 2-second UI responsiveness requirement
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("🔍 *Consulting global live data distribution servers...*")
        
        # Load compiled graph loop architecture
        agent_executor = get_agent()
        
        if agent_executor is None:
            err_msg = "⚠️ All configured API keys are currently out of quota processing limits. Please review your Streamlit secrets pool dashboard inputs."
            response_placeholder.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
        else:
            try:
                # Execute graph logic sequence
                config = {"configurable": {"thread_id": "travel_session_v2"}}
                agent_output = agent_executor.invoke({"messages": [("user", user_input)]}, config=config)
                
                # Extract response text safely
                final_reply = agent_output["messages"][-1].content
                
                response_placeholder.markdown(final_reply)
                st.session_state.messages.append({"role": "assistant", "content": final_reply})
            except Exception as e:
                fallback_error = f"System Error processing request sequence channel: {str(e)}"
                response_placeholder.markdown(fallback_error)
                st.session_state.messages.append({"role": "assistant", "content": fallback_error})
