from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from typing import List
from langchain_core.documents import Document
import uuid
from rag.docstore import SQLiteDocstore

def get_embeddings():
    """Returns the OpenAI Embeddings model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def create_vectorstore(docs: List[Document]):
    """Creates a FAISS vector store with IVF Index from documents."""
    embeddings_model = get_embeddings()
    
    if not docs:
        print("No documents to index.")
        dim = 1536 
        index = faiss.IndexFlatL2(dim)
        return FAISS(embeddings_model, index, InMemoryDocstore(), {})

    print(f"Generating embeddings for {len(docs)} chunks...")
    texts = [d.page_content for d in docs]
    embeddings = embeddings_model.embed_documents(texts)
    embeddings_np = np.array(embeddings).astype('float32')
    
    dim = embeddings_np.shape[1]
    n_samples = embeddings_np.shape[0]
    
    # Calculate nlist
    nlist = int(4 * (n_samples ** 0.5))
    if nlist < 1: nlist = 1
    if n_samples < 39 * nlist:
        nlist = max(1, n_samples // 39)
        if nlist == 0: nlist = 1

    print(f"Creating IVF index with nlist={nlist} for {n_samples} samples...")
    
    # Use Inner Product (Cosine Similarity)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index
    index.train(embeddings_np)
    
    # Add vectors
    index.add(embeddings_np)
    
    # Set nprobe
    index.nprobe = min(10, nlist)
    
    # Use SQLiteDocstore
    docstore = SQLiteDocstore() # Defaults to docstore.db
    index_to_docstore_id = {}
    
    ids = [str(uuid.uuid4()) for _ in docs]
    
    # Add to docstore
    docstore.add({id_: doc for id_, doc in zip(ids, docs)})
    
    # Map index IDs to docstore IDs
    for i, id_ in enumerate(ids):
        index_to_docstore_id[i] = id_
        
    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vectorstore

def save_vectorstore(vectorstore, path: str):
    """Saves the vector store index to disk (docstore is already persistent)."""
    # We only need to save the FAISS index and the index_to_docstore_id mapping
    # FAISS.save_local saves everything, but we want to avoid pickling the SQLiteDocstore connection if possible,
    # or rather, we want to ensure when we load it back, we reconnect.
    # Standard save_local might try to pickle the docstore.
    # Let's see if we can just use save_local and it handles it, or if we need to be careful.
    # SQLite objects are not picklable.
    # So we might need to NOT save the docstore in the pickle, but re-initialize it on load.
    
    # Workaround: FAISS save_local saves the docstore. 
    # We can temporarily replace docstore with None or a dummy before saving?
    # Or better, just save the index and the mapping separately?
    
    # Actually, FAISS.save_local calls `self.index.save(folder_path / "index.faiss")` and pickles the rest.
    # If we pass `docstore` as a SQLiteDocstore, pickle will fail.
    
    # So we should probably just save the index and the mapping manually?
    # Or use `vectorstore.save_local` but ensure `docstore` is not pickled?
    
    # Let's try to just save the index and mapping.
    import pickle
    import os
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    faiss.write_index(vectorstore.index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "index.pkl"), "wb") as f:
        pickle.dump((vectorstore.index_to_docstore_id), f)

def load_vectorstore(path: str):
    """Loads the vector store from disk."""
    import pickle
    import os
    
    embeddings = get_embeddings()
    
    # Load index
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    
    # Load mapping
    with open(os.path.join(path, "index.pkl"), "rb") as f:
        index_to_docstore_id = pickle.load(f)
        
    # Re-initialize docstore
    docstore = SQLiteDocstore()
    
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

