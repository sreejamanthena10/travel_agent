import streamlit as st
import os
import json

# --- FROM WEEK 2/3: Import your LangGraph multi-tool agent brain ---
from agent import get_agent 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration (Week 1 - Undisturbed) ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. Sidebar Setup (Week 1 - Undisturbed) ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- 3. Knowledge Base Loader (Week 1 - Cleaned of invisible characters) ---
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
        
        # 1. Initialize with just the FIRST document to ensure 1:1 length
        vector_db = FAISS.from_documents([all_pages[0]], embeddings)
        
        # 2. Add the remaining documents one-by-one
        if len(all_pages) > 1:
            for page in all_pages[1:]:
                vector_db.add_documents([page])
        
        return vector_db
    return None

# --- FIXED: Use st.cache_resource with a key dependency to avoid initialization loops ---
@st.cache_resource
def get_cached_agent(_key):
    os.environ["GOOGLE_API_KEY"] = _key
    return get_agent()

# Initialize Chat History State Array
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Main App Logic ---
if api_key:
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        vector_db = load_data(api_key)

        if "agent" not in st.session_state:
            st.session_state.agent = get_cached_agent(api_key)

        if vector_db:
            st.write("I can search the web and check the weather for you!")

            # Render past chat messages onto the screen
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # User input layout box
            user_input = st.chat_input("Ask me a question...")

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)

                # Search Week 1 PDFs for matching context
                docs = vector_db.similarity_search(user_input, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                combined_prompt = (
                    f"Use this extracted context from the user's travel documents to help answer if relevant:\n"
                    f"{context}\n\n"
                    f"User Question: {user_input}\n\n"
                    f"Note: If the document context isn't enough, or if you need current information "
                    f"(like real-time weather or web details), use your tools automatically."
                )

                # Get the AI's answer using tools
                with st.chat_message("assistant"):
                    with st.spinner("Using tools to find the answer..."):
                        
                        result = st.session_state.agent.invoke({"messages": [("user", combined_prompt)]})
                        answer = result["messages"][-1].content
                        
                        # --- DOUBLE-CHECKED EXTRACTION FILTER ---
                        if isinstance(answer, str) and ("text" in answer or "signature" in answer):
                            try:
                                # Clean off anomalous prefix markers that break json formatting
                                clean_target = answer.strip().lstrip("0123456789:[] ,\t\n")
                                if not clean_target.startswith("{") and "{" in clean_target:
                                    clean_target = clean_target[clean_target.index("{"):]
                                    
                                parsed_json = json.loads(clean_target)
                                if "text" in parsed_json:
                                    answer = parsed_json["text"]
                            except Exception:
                                # Fallback raw string boundaries split if json parser hits malformed data
                                for anchor in ['"text":"', '"text": "']:
                                    if anchor in answer:
                                        extracted = answer.split(anchor, 1)[1]
                                        for stop_sign in ['","extras"', '",\n"extras"', '"\n"extras"']:
                                            if stop_sign in extracted:
                                                extracted = extracted.split(stop_sign, 1)[0]
                                        answer = extracted.rstrip('"\n\t }]]')
                                        break
                        
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
        else:
            st.error("⚠️ No PDFs found. Make sure they are in data/raw/ on GitHub.")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to begin.")
