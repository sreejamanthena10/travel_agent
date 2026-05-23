
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# 🚀 Added from Week 2/3: Import your LangGraph multi-tool agent
from agent import get_agent 

# --- 1. Page Configuration (Week 1 - Undisturbed) ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. Sidebar Setup (Week 1 - Undisturbed) ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- 3. Knowledge Base Loader (Week 1 - Undisturbed) ---
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
        # Use the newest stable model ID
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        
        # --- THE FIX ---
        # 1. Initialize with just the FIRST document to ensure 1:1 length
        vector_db = FAISS.from_documents([all_pages[0]], embeddings)
        
        # 2. Add the remaining documents one-by-one
        if len(all_pages) > 1:
            for page in all_pages[1:]:
                vector_db.add_documents([page])
        
        return vector_db
    return None

# 🚀 Added from Week 2: Initialize Agent and Chat History State
if "agent" not in st.session_state:
    st.session_state.agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Main App Logic ---
if api_key:
    try:
        vector_db = load_data(api_key)

        if vector_db:
            # Week 2 Text Layout Element
            st.write("I can search the web and check the weather for you!")

            # Week 2: Show past messages from state memory
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Week 2 Text Layout Input Box
            user_input = st.chat_input("Ask me a question...")

            if user_input:
                # 1. Show and save what the user typed (Week 2 Logic)
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)

                # Search for matching local text inside your Week 1 PDFs
                docs = vector_db.similarity_search(user_input, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                # Combine Week 1 Context with the Week 2 Query for the Agent Brain
                prompt = (
                    f"Use this extracted context from the user's travel documents to help answer if relevant:\n"
                    f"{context}\n\n"
                    f"User Question: {user_input}\n\n"
                    f"Note: If the document context isn't enough, or if you need current information "
                    f"(like real-time weather or web details), use your tools automatically."
                )

                # 2. Get the AI's answer using tools (Week 2 Framework + LangGraph Fix)
                with st.chat_message("assistant"):
                    with st.spinner("Using tools to find the answer..."):
                        # Send everything into your active LangGraph engine
                        result = st.session_state.agent.invoke({"messages": [("user", prompt)]})
                        answer = result["messages"][-1].content
                        
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
        else:
            st.error("⚠️ No PDFs found. Make sure they are in data/raw/ on GitHub.")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to begin.")
