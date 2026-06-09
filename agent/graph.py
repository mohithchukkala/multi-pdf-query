from langgraph.graph import END, StateGraph
from agent.state import GraphState
from agent.nodes import (
    retrieve,
    grade_documents,
    transform_query,
    web_search_node,
    generate,
    hallucination_check,
)

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    run_web_search = state["run_web_search"]
    
    if run_web_search:
        # All documents were irrelevant, we will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---CHECK HALLUCINATIONS---")
    # We already ran the check in the node, but here we just return the decision based on the output
    # Actually, the check logic is complex, let's move the logic from the node to here or keep it in the node and return a status.
    # In my node implementation, I returned "useful", "not useful", "not supported".
    # But the node function signature in langgraph usually returns state update.
    # Ah, I made a mistake in `nodes.py`: `hallucination_check` returns a string, not a dict.
    # I should fix `nodes.py` to return state, and put the logic in the conditional edge function.
    # OR, I can keep `hallucination_check` as a function that returns a status, but it needs to be called by a node or edge.
    
    # Let's fix `nodes.py` first to return state, and store the grade in state?
    # Or better, just implement the logic here in the conditional edge.
    
    # Re-reading my nodes.py: `hallucination_check` returns a string. This is NOT a valid node return value if it's a node.
    # It should be a conditional edge function.
    
    # Let's use `hallucination_check` as the conditional function for the edge starting from `generate`.
    # But `generate` returns state.
    
    # So: Generate -> (conditional edge using hallucination_check) -> End or Generate (loop)
    
    # I will import the logic from nodes.py, but I need to refactor nodes.py to separate the logic.
    pass

# Redefining the graph construction
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate", generate)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")

# Conditional edge for hallucination check
# I need to implement the check logic here or import it.
# I'll use a wrapper function here.

from agent.nodes import hallucination_check as check_logic

def check_hallucination_edge(state):
    result = check_logic(state)
    if result == "useful":
        return "useful"
    elif result == "not useful":
        return "not useful"
    else:
        return "not supported"

workflow.add_conditional_edges(
    "generate",
    check_hallucination_edge,
    {
        "useful": END,
        "not useful": "transform_query", # If not useful (doesn't answer question), try searching again?
        "not supported": "generate", # If hallucinated, retry generation? Or maybe retrieve again?
    },
)

# Compile
app = workflow.compile()
