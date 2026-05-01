import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration ---
st.set_page_config(page_title="Travel AI", layout="centered")
st.title("✈️ AI Travel Concierge")

# --- 2. Sidebar Setup ---
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- 3. Knowledge Base Loader ---
@st.cache_resource
def load_data(_key): 
    # Sets the API key inside the cached function
    os.environ["GOOGLE_API_KEY"] = _key
    
    # Path to your PDFs on GitHub
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
        # Using the corrected model name from the 404 error
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return FAISS.from_documents(all_pages, embeddings)
    return None

# --- 4. Main App Logic ---
if api_key:
    try:
        vector_db = load_data(api_key)

        if vector_db:
            query = st.chat_input("Ask a question about your trip:")
            if query:
                with st.chat_message("user"):
                    st.markdown(query)
                
                # Search for matching text in PDFs
                docs = vector_db.similarity_search(query, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                # Generate answer with Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                prompt = f"Use this context: {context}\n\nQuestion: {query}"
                response = llm.invoke(prompt)
                
                with st.chat_message("assistant"):
                    st.markdown(response.content)
        else:
            st.error("⚠️ No PDFs found. Make sure they are in data/raw/ on GitHub.")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👋 Please enter your Gemini API Key in the sidebar to begin.")
