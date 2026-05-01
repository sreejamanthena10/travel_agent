import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- Basic Setup ---
st.set_page_config(page_title="Travel AI")
st.title("✈️ Simple Travel Assistant")

# --- Sidebar for API Key ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize models
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- Load PDF ---
    @st.cache_resource
    def load_data():
        path = "./data/raw"
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.pdf')]
            if files:
                loader = PyPDFLoader(os.path.join(path, files[0]))
                pages = loader.load_and_split()
                return FAISS.from_documents(pages, embeddings)
        return None

    vector_db = load_data()

    if vector_db:
        # --- Simple Chat ---
        query = st.chat_input("Ask about your trip:")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            
            # Find relevant text from PDF
            docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Get response from Gemini
            prompt = f"Context: {context}\n\nQuestion: {query}"
            response = llm.invoke(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(response.content)
    else:
        st.error("⚠️ No PDF found in 'data/raw/'.")
else:
    st.info("👋 Enter your API Key in the sidebar to start.")
