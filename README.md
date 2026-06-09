# Agentic RAG System

An advanced Retrieval Augmented Generation (RAG) system that uses a graph-based agentic architecture to intelligently answer questions. It combines local document retrieval (e.g., SEC filings) with web search, self-correction, and hallucination checks to ensure high-quality, accurate responses.

## Features

- **Agentic Workflow**: Uses a state graph to orchestrate retrieval, grading, and generation.
- **Hybrid Search**: Searches local PDF documents first, falls back to web search if needed.
- **Self-Correction**: Automatically rewrites queries and retries if retrieved documents are irrelevant.
- **Hallucination Checking**: Verifies that generated answers are grounded in facts and actually address the user's question.
- **Dual Interfaces**:
  - **CLI**: Simple command-line tools for ingestion and querying.
  - **Web UI**: A user-friendly Streamlit interface.

## Prerequisites

- Python 3.10+
- OpenAI API Key (for LLM and Embeddings)
- Tavily API Key (for Web Search)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rohith-polisetty/agenticrag.git
    cd agenticrag
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    TAVILY_API_KEY=your_tavily_api_key
    ```

## Usage

### 1. Ingest Documents
Before asking questions, you need to ingest your PDF documents into the vector store.

**CLI:**
```bash
python main.py ingest path/to/your/document.pdf
```

**UI:**
You can also upload files directly through the Streamlit UI.

### 2. Run the Application

**Command Line Interface (CLI):**
Ask a question directly from the terminal:
```bash
python main.py query "What are the risks mentioned in the document?"
```

**Web User Interface (Streamlit):**
Launch the interactive web app:
```bash
streamlit run ui.py
```

## Project Structure

- **`main.py`**: Entry point for CLI commands (ingest, query).
- **`ui.py`**: Streamlit application for the web interface.
- **`agent/`**: Contains the core agent logic.
    - `graph.py`: Defines the LangGraph workflow.
    - `nodes.py`: Implements the steps (retrieve, grade, generate, etc.).
    - `state.py`: Defines the state object passed between nodes.
- **`rag/`**: Handles document processing.
    - `ingest.py`: PDF loading and chunking.
    - `vectorstore.py`: FAISS vector store management.
- **`tools/`**: External tool integrations.
    - `sec_tools.py`: Retrieval from local vector store.
    - `search_tools.py`: Web search integration.
