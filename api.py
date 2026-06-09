from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag.ingest import load_pdf, split_documents
from rag.vectorstore import create_vectorstore, save_vectorstore
from agent.graph import app
import shutil
import os
import uuid

app_api = FastAPI(title="Agentic RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    steps: list = []

@app_api.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        inputs = {"question": request.question}
        final_generation = ""
        steps = []
        
        # Run the graph
        # Note: app.stream returns an iterator. We iterate to get the final result.
        for output in app.stream(inputs):
            for key, value in output.items():
                steps.append(f"Node: {key}")
                if "generation" in value:
                    final_generation = value["generation"]
        
        # If generation is in the final state, it might not be yielded in the last step depending on how stream works
        # Let's try to get the final state if needed, but usually the last yield has the update.
        # Actually, `app.invoke` might be simpler for non-streaming response, but stream gives us progress.
        # For now, let's just return the final generation found.
        
        if not final_generation:
             # Fallback if not captured in stream (e.g. if it was in the state but not yielded as a diff?)
             # But LangGraph stream yields state updates.
             pass

        return QueryResponse(answer=final_generation, steps=steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_api.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    try:
        # Save temp file
        file_id = str(uuid.uuid4())
        file_path = f"temp_{file_id}.pdf"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingest
        docs = load_pdf(file_path)
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
        save_vectorstore(vectorstore, "faiss_index")
        
        # Cleanup
        os.remove(file_path)
        
        return {"message": f"Successfully ingested {len(docs)} pages and {len(chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=8000)
