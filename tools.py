import os
import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun

# --- CORE INNER LIVE EXECUTION UTILITY ---
def run_live_web_lookup(search_query: str) -> str:
    """Helper function to cleanly scrape live data directly from the active web stream."""
    try:
        search_engine = DuckDuckGoSearchRun()
        return str(search_engine.run(search_query))
    except Exception as e:
        return f"Live web data stream temporarily unavailable: {str(e)}"

def run_pdf_rag_search(query: str) -> str:
    """Helper function to execute RAG similarity searches over local travel documents."""
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


# --- 4 REQUIRED LIVE AGENT TOOL VECTOR ENTRY CORES ---

@tool
def search_flights(query: str) -> str:
    """
    Searches the live web for completely real-time flight options, active airline carrier schedules, 
    and direct route pricing tables matching the user's destination constraints.
    """
    # Check local RAG context index files first
    local_doc_result = run_pdf_rag_search(query)
    if "No local travel documents" not in local_doc_result and local_doc_result.strip():
        return local_doc_result
    
    # Execute a live lookup on the web for real-time airline options
    return run_live_web_lookup(f"current flight schedules airline routes ticket pricing metrics for {query} 2026")

@tool
def search_hotels(query: str) -> str:
    """
    Locates verified premium accommodations, real physical lodging options, and active stay pricing 
    inside specific geographic location parameters.
    """
    local_doc_result = run_pdf_rag_search(query)
    if "No local travel documents" not in local_doc_result and local_doc_result.strip():
        return local_doc_result
        
    return run_live_web_lookup(f"verified actual hotels accommodations stay options pricing details inside {query}")

@tool
def get_weather(query: str) -> str:
    """
    Fetches genuine real-time meteorological conditions, active temperature readings, and localized 
    regional forecast tables.
    """
    return run_live_web_lookup(f"current weather temperature degrees meteorological report for {query}")

@tool
def plan_itinerary(query: str) -> str:
    """
    Assembles customized, highly scannable day-by-day sightseeing timelines, tracking hidden tourist 
    landmarks and local travel events.
    """
    local_doc_result = run_pdf_rag_search(query)
    if "No local travel documents" not in local_doc_result and local_doc_result.strip():
        return local_doc_result
        
    return run_live_web_lookup(f"comprehensive day travel itinerary sightseeing landmarks path timeline for {query}")
