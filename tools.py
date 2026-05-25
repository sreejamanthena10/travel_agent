import os
import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun

# --- CORE INNER EXECUTION ENGINE ---
def run_live_web_search(query: str) -> str:
    """Helper function to execute real-time web lookups safely."""
    try:
        search_engine = DuckDuckGoSearchRun()
        return str(search_engine.run(query))
    except Exception as e:
        return f"Live lookup temporarily unavailable: {str(e)}"

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


# --- REQUIRED AGENT COUPLING GATEWAY (4 ALIGNED TOOL FUNCTIONS) ---

@tool
def search_flights(query: str) -> str:
    """
    Searches the live web for global travel information, airline schedules, plane metrics, 
    and route prices matching the user's destination parameters.
    """
    # First check if the answer exists inside your uploaded documents
    local_doc_result = run_pdf_rag_search(query)
    if "No local travel documents" not in local_doc_result and local_doc_result.strip():
        return local_doc_result
    # Otherwise, fetch real-time true details from the live web engine
    return run_live_web_search(f"flights schedule carrier price matrix {query}")

@tool
def search_hotels(query: str) -> str:
    """
    Locates verified premium accommodations, tier-priced stay matrices, and rating features 
    at specific user coordinates.
    """
    local_doc_result = run_pdf_rag_search(query)
    if "No local travel documents" not in local_doc_result and local_doc_result.strip():
        return local_doc_result
    return run_live_web_search(f"hotels accommodations stay pricing rate details {query}")

@tool
def get_weather(query: str) -> str:
    """
    Fetches official current atmospheric conditions, temperature trends, and regional 
    climate forecast indicators.
    """
    return run_live_web_search(f"current weather temperature forecast metrics {query}")

@tool
def plan_itinerary(query: str) -> str:
    """
    Builds customized, scannable day-by-day sightseeing blueprints, tracking hidden tourist 
    landmarks and local travel events.
    """
    local_doc_result = run_pdf_rag_search(query)
    if "No local travel documents" not in local_doc_result and local_doc_result.strip():
        return local_doc_result
    return run_live_web_search(f"travel itinerary tourist spots sightseeing guide {query}")
