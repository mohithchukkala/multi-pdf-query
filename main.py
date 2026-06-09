import os
from dotenv import load_dotenv

load_dotenv()

import sys
from rag.ingest import load_pdf, split_documents
from rag.vectorstore import create_vectorstore, save_vectorstore
from agent.graph import app

def ingest(file_path: str):
    print(f"Loading PDF from {file_path}...")
    docs = load_pdf(file_path)
    print(f"Loaded {len(docs)} pages.")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore, "faiss_index")
    print("Vector store saved to 'faiss_index'.")

def run_agent(question: str):
    print(f"Running agent for question: {question}")
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            # print(value) # Optional: print state
    print("\n--- Final Answer ---")
    print(value["generation"])

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest <pdf_path> | query <question>]")
        return

    command = sys.argv[1]
    
    if command == "ingest":
        if len(sys.argv) < 3:
            print("Please provide a PDF path.")
            return
        ingest(sys.argv[2])
    elif command == "query":
        if len(sys.argv) < 3:
            print("Please provide a question.")
            return
        question = " ".join(sys.argv[2:])
        run_agent(question)
    else:
        print("Unknown command.")

