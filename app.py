# --- 4. Caching Layers for Maximum Performance Optimization ---
@st.cache_resource
def load_data(_key): 
    os.environ["GOOGLE_API_KEY"] = _key
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
        # PRODUCTION MIGRATION FIX: Swapped deprecated text-embedding-004 with active gemini-embedding-001
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = FAISS.from_documents([all_pages[0]], embeddings)
        if len(all_pages) > 1:
            for page in all_pages[1:]:
                vector_db.add_documents([page])
        return vector_db
    return None
