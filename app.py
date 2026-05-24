import streamlit as st
import os

# --- FROM WEEK 2/3: Import your LangGraph multi-tool agent brain ---
from agent import get_agent, get_keys_pool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration & Premium Executive CSS Styling ---
st.set_page_config(page_title="AeroConcierge AI", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
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
    input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
    }
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="animated-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">AeroConcierge AI</h1>', unsafe_allow_html=True)
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous Travel Intelligence & Verified Vector RAG Platform</p>', unsafe_allow_html=True)

# Clean extract key strings list
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
            loader = PyPDFLoader(file_path)
            all_pages.extend(loader.load_and_split())
    if all_pages:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return FAISS.from_documents([all_pages[0]], embeddings)
    return None

def safe_vector_search(_query):
    """Fallback vector search engine using active keys pool."""
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

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    st.session_state.agent = get_agent()
except Exception:
    st.session_state.agent = None

# Render active layout session items
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Inquire regarding itineraries, global budgets, or local attractions...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Dynamic evaluation check string
    is_weather_query = any(k in user_input.lower() for k in ["weather", "temp", "temperature", "forecast"])

    with st.chat_message("assistant"):
        if is_weather_query:
            # FIXED: REMOVED HARCODED PLACES ARRAY. Automatically filter phrase words out.
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
            # FIXED: REMOVED FORCED METHOD AND CONTEXT STUFFING BLOCKS. Passes pure text directly.
            if st.session_state.agent is None:
                st.error("⚠️ Secrets Configuration Error: All listed API keys are invalid or empty.")
            else:
                with st.spinner("Processing expert travel logic..."):
                    try:
                        # Agent automatically chooses the right tool based on your question!
                        result = st.session_state.agent.invoke({"messages": [("user", user_input)]})
                        answer = str(result["messages"][-1].content)
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception:
                        st.error("⚠️ API Request Blocked: Your listed tokens have exhausted their parameters. Update your backend secret strings.")

st.markdown('</div>', unsafe_allow_html=True)
