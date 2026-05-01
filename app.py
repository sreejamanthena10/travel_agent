import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Config ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. API Key Sidebar ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- 3. Knowledge Base Loader ---
@st.cache_resource
def load_data(_key): 
    # Directly set the environment variable inside the cached function
    os.environ["GOOGLE_API_KEY"] = _key
    
    # Locate the PDF folder
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
        # Initialize embeddings precisely when needed
     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # This line converts text to vectors using the API key
        return FAISS.from_documents(all_pages, embeddings)
    return None

# --- 4. Main Logic ---
if api_key:
    try:
        # Pass the key to the loader
        vector_db = load_data(api_key)

        if vector_db:
            query = st.chat_input("Ask about your travel docs:")
            if query:
                with st.chat_message("user"):
                    st.markdown(query)
                
                # Context Search
                docs = vector_db.similarity_search(query, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                # Chat Model
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                prompt = f"Context: {context}\n\nQuestion: {query}"
                response = llm.invoke(prompt)
                
                with st.chat_message("assistant"):
                    st.markdown(response.content)
        else:
            st.error("⚠️ No PDF files found in 'data/raw/'. Please upload them to GitHub.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Try rebooting the app from the 'Manage app' menu.")
else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to start.")
