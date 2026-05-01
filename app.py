import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Basic Page Config ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ Simple Travel Assistant")

# --- 2. Sidebar for API Key ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize models
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 3. Fail-Safe PDF Loader ---
    @st.cache_resource
    def load_data():
        # DYNAMIC PATH: This looks for the 'data/raw' folder relative to app.py
        base_path = os.path.dirname(__file__)
        data_folder = os.path.join(base_path, "data", "raw")
        
        if os.path.exists(data_folder):
            # Get a list of all PDFs in that folder
            files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
            
            if files:
                # Load the first PDF found regardless of its name
                loader = PyPDFLoader(os.path.join(data_folder, files[0]))
                pages = loader.load_and_split()
                return FAISS.from_documents(pages, embeddings)
        return None

    vector_db = load_data()

    if vector_db:
        # --- 4. Simple Chat Interface ---
        query = st.chat_input("Ask about your trip:")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Search PDF for answer
            docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Simple prompt logic
            prompt = f"Using this info: {context}\n\nAnswer this: {query}"
            response = llm.invoke(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(response.content)
    else:
        st.error(f"⚠️ No PDF found! Please ensure your file is in a folder named 'data/raw' on GitHub.")
else:
    st.info("👋 Enter your Gemini API Key in the sidebar to start.")
