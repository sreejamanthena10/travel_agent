import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Travel Concierge", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. API Key Setup ---
# Enter your Gemini API Key in the sidebar
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize Gemini & Embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 3. Knowledge Base Logic ---
    @st.cache_resource
    def load_kb():
        # Looks for your PDF in the data/raw folder
        folder = "./data/raw"
        if os.path.exists(folder):
            pdfs = [f for f in os.listdir(folder) if f.endswith('.pdf')]
            if pdfs:
                # Load the first PDF found for simplicity
                loader = PyPDFLoader(os.path.join(folder, pdfs[0]))
                pages = loader.load_and_split()
                return FAISS.from_documents(pages, embeddings)
        return None

    vector_db = load_kb()

    if vector_db:
        # --- 4. Chat Interface ---
        query = st.chat_input("Ask a question about your travel documents:")
        
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Simple, stable QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever()
            )
            
            with st.chat_message("assistant"):
                response = qa_chain.run(query)
                st.markdown(response)
    else:
        st.error("⚠️ No PDF files found in `data/raw/`. Please upload a PDF to GitHub.")
else:
    st.info("👋 Welcome! Please enter your Gemini API Key in the sidebar to start.")
