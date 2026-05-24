import streamlit as st
import os
import json

# --- FROM WEEK 2/3: Import your LangGraph multi-tool agent brain ---
from agent import get_agent 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration & Premium Styling ---
st.set_page_config(page_title="AI Travel Concierge", layout="centered")

# Custom CSS Injection for high-end UI/UX, premium color combinations, and fade-in animations
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global Fade-In Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animated-container {
        animation: fadeIn 0.8s ease-out forwards;
    }
    
    /* Professional Header Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Chat Box and Styling Elements */
    input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    .stAlert {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid #475569 !important;
        color: #f8fafc !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Main Premium App Layout ---
st.markdown('<div class="animated-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">✈️ AI TRAVEL CONCIERGE</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your documents, web resources, and real-time environment data connected seamlessly.</p>', unsafe_allow_html=True)

# --- 3. Secure Production Key Injection ---
# Pulls directly from your secure Streamlit Dashboard Cloud settings secretly
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("⚠️ Environment Configuration Missing: Please set 'GEMINI_API_KEY' in your Streamlit Advanced Settings.")
    st.stop()

# --- 4. Knowledge Base Loader ---
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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
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

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. Application Core Execution Loop ---
try:
    os.environ["GOOGLE_API_KEY"] = api_key
    vector_db = load_data(api_key)

    if "agent" not in st.session_state:
        st.session_state.agent = get_cached_agent(api_key)

    if vector_db:
        st.write("✨ Intelligence matrix fully synchronized. Ask me anything!")

        # Render chat history with smooth styles
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat interface panel
        user_input = st.chat_input("Inquire regarding your upcoming itineraries or destination weather...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            docs = vector_db.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            combined_prompt = (
                f"Use this extracted context from the user's travel documents to help answer if relevant:\n"
                f"{context}\n\n"
                f"User Question: {user_input}\n\n"
                f"Note: If the document context isn't enough, or if you need current information "
                f"(like real-time weather or web details), use your tools automatically."
            )

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data vectors and computing response..."):
                    
                    result = st.session_state.agent.invoke({"messages": [("user", combined_prompt)]})
                    last_message = result["messages"][-1]
                    answer = ""
                    
                    if hasattr(last_message, "content") and isinstance(last_message.content, list):
                        extracted_chunks = []
                        for chunk in last_message.content:
                            if isinstance(chunk, dict) and "text" in chunk:
                                extracted_chunks.append(chunk["text"])
                            elif isinstance(chunk, str):
                                extracted_chunks.append(chunk)
                        answer = "\n".join(extracted_chunks)
                    elif hasattr(last_message, "content"):
                        answer = str(last_message.content)
                    else:
                        answer = str(last_message)
                    
                    if '"text":' in answer or '"signature":' in answer:
                        for anchor in ['"text":"', '"text": "']:
                            if anchor in answer:
                                sliced_data = answer.split(anchor, 1)[1]
                                for terminator in ['","extras"', '",\n"extras"', '"\n"extras"']:
                                    if terminator in sliced_data:
                                        sliced_data = sliced_data.split(terminator, 1)[0]
                                answer = sliced_data.rstrip('"\n\t }]]')
                                break

                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                        
    else:
        st.error("⚠️ Document Vector Synchronization Failed: Verify that your itinerary PDFs reside within 'data/raw/' on GitHub.")
        
except Exception as e:
    st.error(f"❌ System Exception Encountered: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
