import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. Basic Page Config ---
st.set_page_config(page_title="AI Travel Assistant", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. API Key Sidebar ---
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize models
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 3. Knowledge Base ---
    @st.cache_resource
    def load_kb():
        # Path must match your GitHub folder structure
        folder = "./data/raw"
        if os.path.exists(folder):
            pdfs = [f for f in os.listdir(folder) if f.endswith('.pdf')]
            if pdfs:
                # Load the first PDF found
                loader = PyPDFLoader(os.path.join(folder, pdfs[0]))
                pages = loader.load_and_split()
                return FAISS.from_documents(pages, embeddings)
        return None

    vector_db = load_kb()

    if vector_db:
        # --- 4. Chat logic ---
        query = st.chat_input("Ask about your trip (e.g., flight details):")
        
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Use the most stable QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever()
            )
            
            with st.chat_message("assistant"):
                response = qa_chain.run(query)
                st.markdown(response)
    else:
        st.error("⚠️ No PDFs found in `data/raw/`. Please check your GitHub folders.")
else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to start.")
