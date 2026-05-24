import os
import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the free DuckDuckGo Search Engine Utility
try:
    search_engine = DuckDuckGoSearchRun()
except Exception:
    search_engine = None

@tool
def google_travel_search(query: str) -> str:
    """
    Searches the live web for global travel information, flight details, hotel pricing, 
    and famous landmark updates all over the world. Use this for general global destinations.
    """
    if search_engine:
        try:
            return str(search_engine.run(query))
        except Exception as e:
            return f"Live search temporarily unavailable: {str(e)}"
    return "Search infrastructure uninitialized."

@tool
def search_local_travel_documents(query: str) -> str:
    """
    Searches local uploaded PDF travel documents, local vouchers, and specific destination itineraries.
    Use this tool ONLY when the user asks about specific personal plans, uploads, or schedules matching local files.
    """
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "data", "raw")
    all_pages = []
    
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        for f in files:
            file_path = os.path.join(data_folder, f)
            try:
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load_and_split())
            except Exception:
                continue
            
    if all_pages:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_db = FAISS.from_documents(all_pages, embeddings)
            docs = vector_db.similarity_search(query, k=2)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Error reading internal document index: {str(e)}"
    
    return "No local travel documents found in the database directory."

# Export the tools clearly for agent.py
my_tools = [google_travel_search, search_local_travel_documents]
