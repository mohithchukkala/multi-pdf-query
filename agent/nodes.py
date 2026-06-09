from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from agent.state import GraphState
from tools.sec_tools import retrieve_documents, init_retriever
from tools.search_tools import web_search
from tools.market_tools import get_stock_price

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Data Models for Structured Output ---

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to check if the answer resolves the question."""
    binary_score: str = Field(description="Answer resolves the question, 'yes' or 'no'")

# --- Nodes ---

def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore.
    """
    print("---RETRIEVE---")
    question = state["question"]
    search_query = state.get("search_query", question) # Use rewritten query if available

    # Initialize retriever if needed (this might be better in main setup)
    init_retriever()
    
    documents = retrieve_documents.invoke(search_query)
    return {"documents": [documents], "question": question}

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    run_web_search = False
    
    for doc in documents:
        score = grader.invoke({"question": question, "document": doc})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            run_web_search = True # If any doc is irrelevant, we might want to rewrite or search web
            
    # If no relevant docs found, we definitely need to rewrite/search
    if not filtered_docs:
        run_web_search = True
        
    return {"documents": filtered_docs, "question": question, "run_web_search": run_web_search}

def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    
    question_rewriter = re_write_prompt | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
    
    return {"documents": documents, "search_query": better_question}

def web_search_node(state: GraphState):
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    search_query = state.get("search_query", question)
    
    docs = web_search.invoke(search_query)
    web_results = f"\nWeb Search Result: {docs}"
    
    documents = state["documents"]
    documents.append(web_results)
    
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """
    Generate answer using the vectorstore documents.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("human", "Question: {question} \n\n Context: {context} \n\n Answer:"),
        ]
    )
    
    rag_chain = prompt | llm | StrOutputParser()
    
    # Combine docs
    context = "\n\n".join(documents)
    
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def hallucination_check(state: GraphState):
    """
    Checks for hallucinations.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    grader = hallucination_prompt | structured_llm_grader
    score = grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check if it answers the question
        structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."""
        
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_answer),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        grader_answer = answer_prompt | structured_llm_grader_answer
        score_answer = grader_answer.invoke({"question": question, "generation": generation})
        
        if score_answer.binary_score == "yes":
             print("---DECISION: GENERATION ADDRESSES QUESTION---")
             return "useful"
        else:
             print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
             return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
