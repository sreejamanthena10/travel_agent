!pip install streamlit
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Page Setup
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ Travel Concierge")

# 2. Key Handling
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize Models
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. Simple Knowledge Base
    @st.cache_resource
    def load_kb():
        path = "./data/raw"
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.pdf')]
            if files:
                # Load the first PDF found
                loader = PyPDFLoader(os.path.join(path, files[0]))
                # split docs into chunks automatically
                pages = loader.load_and_split() 
                return FAISS.from_documents(pages, embeddings)
        return None

    vector_db = load_kb()

    if vector_db:
        # 4. Chat Interface
        query = st.chat_input("Ask a question about your trip:")
        
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Use the most stable QA chain available
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever()
            )
            
            with st.chat_message("assistant"):
                response = qa_chain.run(query)
                st.markdown(response)
    else:
        st.error("⚠️ No PDF found in 'data/raw/'. Check your GitHub folders!")
else:
    st.info("Please enter your API Key in the sidebar.")
