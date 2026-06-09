from langchain_core.tools import tool
from rag.vectorstore import load_vectorstore

# Global variable to hold the vectorstore, initialized later
vectorstore = None

def init_retriever(index_path: str = "faiss_index"):
    global vectorstore
    try:
        vectorstore = load_vectorstore(index_path)
    except Exception:
        print("Vector store not found. Please ingest data first.")
        vectorstore = None

@tool
def retrieve_documents(query: str):
    """
    Retrieves relevant documents from the SEC filings vector store.
    Args:
        query: The query string to search for.
    """
    global vectorstore
    if vectorstore is None:
        return "Vector store is not initialized. No documents found."
    
    # Threshold for Cosine Similarity (0 to 1). 
    # 0.3 is a conservative starting point; adjust as needed.
    THRESHOLD = 0.3
    
    # search_kwargs={"k": 5} to get more candidates before filtering
    results = vectorstore.similarity_search_with_score(query, k=5)
    
    # Filter by score (Higher is better for Cosine/Inner Product)
    filtered_docs = [doc for doc, score in results if score >= THRESHOLD]
    
    if not filtered_docs:
        return "No relevant documents found (score below threshold)."
        
    return "\n\n".join([doc.page_content for doc in filtered_docs])
