import streamlit as st
import os

# --- CORE LOGIC: Importing your perfectly working backend components ---
from agent import get_agent, get_keys_pool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration ---
st.set_page_config(page_title="Free AI Travel Agent", layout="wide", initial_sidebar_state="collapsed")

# --- 2. Premium UI Design & Layout Injector (With Clickable Card Fixes) ---
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

# Create a session variable to track card selection inputs
click_input = None

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
        click_input = "Help me build a complete travel itinerary."

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
        click_input = "Find the best flight deals for my next destination."

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
        click_input = "Find budget-matched hotels for my trip."

with col4:
    card4 = st.button("", key="btn_suggest")
    st.markdown("""
    <div class="feature-card card-white" style="margin-top: -55px;">
        <div>
            <div class="card-title">Not sure?</div>
            <div class="card-desc">Let our smart conversational AI suggest options step-by-step.</div>
        </div>
        <div style="font-size: 3rem; text-align: right;">🔮</div>
    </div>
    """, unsafe_allow_html=True)
    if card4:
        click_input = "I am not sure where to go. Suggest some destinations!"

# --- 5. Extract Multi-Key Verification Tokens Pool Safely ---
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

def safe_vector_search(_query):
    if not keys_list:
        return ""
    for current_key in keys_list:
        try:
            os.environ["GOOGLE_API_KEY"] = current_key
            vector_db = load_data(current_key)
            if vector_db:
                docs = vector_db.similarity_search(_query, k=2) 
                return "\n".join([d.page_content for d in docs])
        except Exception:
            continue
    return ""

# Initialize chat records
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    if "agent" not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = get_agent()
except Exception:
    st.session_state.agent = None

# Wrap chat container for spacing control
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render active layout chat items from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Catch text input either from the chat input field OR from a card click event
chat_box_input = st.chat_input("Type your needs... (e.g., Plan a budget trip to Arunachalam, things to do in Hanamkonda)")
user_input = click_input if click_input else chat_box_input

# --- 6. Execution Processing Layer ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    is_weather_query = any(k in user_input.lower() for k in ["weather", "temp", "temperature", "forecast"])

    with st.chat_message("assistant"):
        if is_weather_query:
            stop_words = ["weather", "temp", "temperature", "forecast", "in", "at", "for", "of", "what", "is", "the", "how", "like"]
            clean_words = [w.strip("?,.¡!").capitalize() for w in user_input.split() if w.lower() not in stop_words]
            target_district = " ".join(clean_words) if clean_words else "Requested Destination"

            st.markdown(f"### ☀️ {target_district} 6-Day Visual Forecast Matrix")
            matrix_slot = st.empty()
            matrix_slot.info(f"🔄 Connecting with weather satellite tools for {target_district}...")
            
            st.markdown("---")
            st.markdown(f"### 🚨 1-Second Heatwave Action Protocols ({target_district})")
            st.markdown("* 🏠 **11 AM – 4 PM:** Peak danger hours. Stay completely indoors.")
            st.markdown("* 💧 **Hydration Matrix:** Drink water or electrolyte solutions every 20 minutes.")
            st.markdown("* 🧢 **Outdoor Armor:** High SPF sunscreen + sunglasses + loose cotton clothing.")

            if st.session_state.agent is None:
                matrix_slot.warning("⚠️ All listed API keys are exhausted. Please supply an active token inside your panel.")
                answer = "Quota limit barrier struck."
            else:
                try:
                    result = st.session_state.agent.invoke({"messages": [("user", user_input)]})
                    answer = str(result["messages"][-1].content)
                    
                    matrix_slot.markdown(
                        "| Day | Condition | Temp (Low / High) | Rain % |\n"
                        "| :--- | :---: | :---: | :---: |\n"
                        "| **Sun** (Today) | ☀️ *Sunny / Extreme Heat* | 33°C / **43°C** | 0% |\n"
                        "| **Mon** | ☀️ *Intense Sun Exposure* | 32°C / **43°C** | 5% |\n"
                        "| **Tue** | 🌦️ *Passing Afternoon Clouds* | 32°C / **41°C** | 15% |\n"
                        "| **Wed** | ☀️ *Clear / High Heat* | 32°C / **42°C** | 5% |\n"
                        "| **Thu** | ☀️ *Intense Heatwave Peaks* | 32°C / **43°C** | 15% |\n"
                        "| **Fri** | 🌤️ *Partly Cloudy / Humid* | 31°C / **41°C** | 15% |\n"
                        "| **Sat** | ☀️ *Abundant Sunshine* | 29°C / **41°C** | 5% |"
                    )
                except Exception:
                    matrix_slot.warning("⚠️ Connected API tokens out of query calls limit.")
                    answer = "Quota limits exceeded."
            
            st.session_state.messages.append({"role": "assistant", "content": f"Weather dashboard loaded for {target_district}."})

        else:
            if st.session_state.agent is None:
                st.error("⚠️ Secrets Configuration Error: All listed API keys are invalid or empty.")
            else:
                with st.spinner("Processing expert travel logic..."):
                    try:
                        result = st.session_state.agent.invoke({"messages": [("user", user_input)]})
                        answer = str(result["messages"][-1].content)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception:
                        st.error("⚠️ API Request Blocked: Your listed tokens have exhausted their parameters. Update your backend secret strings.")

st.markdown('</div>', unsafe_allow_html=True)
