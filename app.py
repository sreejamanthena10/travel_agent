import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Setup ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. API Key Sidebar ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize models
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 3. Knowledge Base Loader ---
    @st.cache_resource
    def load_data():
        # We try the local GitHub path first, then the Colab path
        possible_files = [
            "./data/raw/Hotel_Booking_Confirmation.pdf",
            "./data/raw/Hyderabad_Travel_Brochure (1).pdf",
            "/content/Hotel_Booking_Confirmation.pdf",
            "/content/Hyderabad_Travel_Brochure (1).pdf"
        ]
        
        all_pages = []
        for file_path in possible_files:
            if os.path.exists(file_path):
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load_and_split())
        
        if all_pages:
            return FAISS.from_documents(all_pages, embeddings)
        return None

    vector_db = load_data()

    if vector_db:
        # --- 4. Chat Interface ---
        query = st.chat_input("Ask about your hotel or Hyderabad trip:")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Simple RAG search
            docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            prompt = f"Use this context to answer: {context}\n\nQuestion: {query}"
            response = llm.invoke(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(response.content)
    else:
        st.error("⚠️ Could not find your PDF files. Please check the 'data/raw' folder on GitHub.")
else:
    st.info("👋 Enter your Gemini API Key in the sidebar to start.")
