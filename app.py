import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Basic Page Config ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. Sidebar for API Key ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- 3. THE FIXED FUNCTION (Paste this here) ---
@st.cache_resource
def load_data(_api_key): 
    # Force the environment variable inside the cached function
    os.environ["GOOGLE_API_KEY"] = _api_key
    
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "data", "raw")
    
    all_pages = []
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        for f in files:
            loader = PyPDFLoader(os.path.join(data_folder, f))
            all_pages.extend(loader.load_and_split())
            
    if all_pages:
        # Initialize embeddings INSIDE the function for stability
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_documents(all_pages, embeddings)
    return None

# --- 4. Main App Logic ---
if api_key:
    # We pass the api_key as an argument so the cache knows to use it
    vector_db = load_data(api_key)

    if vector_db:
        query = st.chat_input("Ask about your trip:")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Simple RAG search
            docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Initialize Chat Model
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            prompt = f"Use this context to answer: {context}\n\nQuestion: {query}"
            response = llm.invoke(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(response.content)
    else:
        st.error("⚠️ No PDF found in 'data/raw/'.")
else:
    st.info("👋 Enter your API Key in the sidebar to start.")
