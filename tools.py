from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os

@tool
def search_local_travel_documents(query: str) -> str:
    """
    Searches local uploaded PDF travel documents, vouchers, and specific destination itineraries.
    Use this tool when the user asks about specific plans, uploads, or schedules matching local files.
    """
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
        # Use the same embedding model defined in your environment
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = FAISS.from_documents(all_pages, embeddings)
        docs = vector_db.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])
    
    return "No local travel documents found."

# Make sure to append search_local_travel_documents to your my_tools list!
# my_tools = [google_search, search_local_travel_documents]
