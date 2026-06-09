from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        search_query: query used for retrieval (can be rewritten)
        run_web_search: whether to run web search
    """
    question: str
    generation: str
    documents: List[str]
    search_query: str
    run_web_search: bool
