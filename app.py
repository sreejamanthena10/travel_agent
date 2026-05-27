import streamlit as st
from agent import get_agent

# --- 1. System Page Configurations ---
st.set_page_config(page_title="Free AI Travel Agent", page_icon="✈️", layout="wide")

# --- 2. Initialize Persistent Session States ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# --- 3. Header Theme Controller (Manual System-Independent Toggle) ---
col_space, col_toggle = st.columns([8, 2])
with col_toggle:
    current_theme = st.session_state.theme
    toggle_label = "🌙 Night Sky Mode" if current_theme == "light" else "☀️ Bright Day Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.theme = "dark" if current_theme == "light" else "light"
        st.rerun()

# --- 4. Dynamic Theme-Independent CSS Engine (Matches "Screenshot (94)_2.jpg") ---
if st.session_state.theme == "dark":
    BG_STYLE = "radial-gradient(circle at 50% 50%, #1e1b4b 0%, #111827 100%)"
    TXT_MAIN = "#ffffff"
    TXT_MUTED = "#94a3b8"
    TXT_ORANGE = "#ff7a33"
    
    # 4 Cards Styling Matrix (Dark Mode)
    CARD_1_BG = "#2e2a14"       # Muted Yellow-Gold
    CARD_2_BG = "#1e293b"       # Soft Light Blue
    CARD_3_BG = "#1e3a8a"       # Deep Blue
    CARD_4_BG = "#1f2937"       # Charcoal Gray
    CARD_BORDER = "#374151"
    FORCE_FONT = "#ffffff"
else:
    BG_STYLE = "radial-gradient(circle at 50% 50%, #fee2e2 0%, #fae8ff 35%, #f5f3ff 65%, #e0f2fe 100%)"
    TXT_MAIN = "#1e293b"
    TXT_MUTED = "#64748b"
    TXT_ORANGE = "#ea580c"      # Warm Orange
    
    # 4 Cards Styling Matrix (Light Mode from Screenshot (94)_2.jpg)
    CARD_1_BG = "#fef08a"       # Pale Yellow
    CARD_2_BG = "#dbeafe"       # Soft Light Blue
    CARD_3_BG = "#bfdbfe"       # Medium Blue
    CARD_4_BG = "#ffffff"       # Pure White
    CARD_BORDER = "#e2e8f0"
    FORCE_FONT = "#1e293b"

CSS_SHEET = f"""
<style>
    /* Force main app container background */
    .stApp {{
        background: {BG_STYLE} !important;
        color: {TXT_MAIN} !important;
    }}
    
    /* Center and format titles matching the reference screenshot image layout */
    .hero-container {{
        text-align: center;
        padding: 1.5rem 0;
    }}
    .hero-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {TXT_ORANGE} !important;
        margin-bottom: 0.5rem;
    }}
    .hero-subtitle {{
        font-size: 1.2rem;
        font-weight: 500;
        color: {TXT_MUTED} !important;
        margin-bottom: 0.5rem;
    }}
    .hero-small {{
        font-size: 0.95rem;
        color: {TXT_MUTED} !important;
        margin-bottom: 2rem;
    }}
    
    /* Exact card container sizing and positioning layouts */
    .ui-card {{
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        padding: 1.8rem;
        min-height: 220px;
        position: relative;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .card-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TXT_MAIN} !important;
        margin-bottom: 0.8rem;
    }}
    .card-desc {{
        font-size: 0.95rem;
        color: {TXT_MUTED} !important;
        line-height: 1.5;
    }}
    .card-icon {{
        font-size: 2.2rem;
        text-align: right;
        margin-top: auto;
    }}
    
    /* Absolute Font Color Overrides across all Chat and Markdown tables */
    .stChatMessage, .stChatMessage p, .stChatMessage div, .stChatMessage span,
    div[data-testid="stMarkdownContainer"] p, td, th, table, tr, li, ul, ol {{
        color: {FORCE_FONT} !important;
    }}
    table {{
        background-color: {CARD_4_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
    }}
</style>
"""
st.markdown(CSS_SHEET, unsafe_allow_html=True)

# --- 5. Main Hero Text Section (Matches Title Text Exactly) ---
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Begin Your Next Adventure 🎈</div>
    <div class="hero-subtitle">Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, and I'll make your ideal plan! Happy Travels! ✈️</div>
    <div class="hero-small">Start by choosing priority service or just describing your needs below!</div>
</div>
""", unsafe_allow_html=True)

# --- 6. Four Custom Cards Layout (Matches Content & Layout of Screenshot (94)_2.jpg) ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="ui-card" style="background-color: {CARD_1_BG};">
        <div>
            <div class="card-title">Build Itinerary</div>
            <div class="card-desc">Tailored completely for your preferences and days.</div>
        </div>
        <div class="card-icon">📍</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="ui-card" style="background-color: {CARD_2_BG};">
        <div>
            <div class="card-title">Find Flights</div>
            <div class="card-desc">Smart deals tracked across multiple global sources.</div>
        </div>
        <div class="card-icon">📅</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="ui-card" style="background-color: {CARD_3_BG};">
        <div>
            <div class="card-title">Find Hotels</div>
            <div class="card-desc">Perfect accommodation metrics matched to your needs.</div>
        </div>
        <div class="card-icon">🏨</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="ui-card" style="background-color: {CARD_4_BG};">
        <div>
            <div class="card-title">Not sure?</div>
            <div class="card-desc">Let our smart conversational AI suggest options step-by-step.</div>
        </div>
        <div class="card-icon">🔮</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr style='border-top: 1px solid var(--stBorderColor);'><br>", unsafe_allow_html=True)

# --- 7. Chat Feed Node ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 8. AI Processing Pipeline Engine ---
if user_input := st.chat_input("Describe your ideal destination journey or asking criteria here..."):
    # Append input immediately to hit sub-2 second responsive standard
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("🔍 *Consulting global live data distribution servers...*")
        
        # Load agent compilation graph safely 
        agent_executor = get_agent()
        
        if agent_executor is None:
            err_text = "⚠️ Token stream parsing failure. Verify that your GEMINI_API_KEYS are saved perfectly in your secrets pool dashboard entries."
            response_placeholder.markdown(err_text)
            st.session_state.messages.append({"role": "assistant", "content": err_text})
        else:
            try:
                # Dispatch query loop parameters across LangGraph React components
                config = {"configurable": {"thread_id": "live_travel_suite_v3"}}
                agent_output = agent_executor.invoke({"messages": [("user", user_input)]}, config=config)
                
                # Fetch final reply content accurately regardless of string complexity or prompt length
                final_reply = agent_output["messages"][-1].content
                
                response_placeholder.markdown(final_reply)
                st.session_state.messages.append({"role": "assistant", "content": final_reply})
            except Exception as e:
                fallback_error = f"Processing Exception: Could not link data tools. Check backend logs: {str(e)}"
                response_placeholder.markdown(fallback_error)
                st.session_state.messages.append({"role": "assistant", "content": fallback_error})
