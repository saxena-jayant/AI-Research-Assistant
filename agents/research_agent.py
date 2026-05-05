from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from tools.search_tool import get_search_tool
from utils.pinecone_db import PineconeManager
from dotenv import load_dotenv
import os

load_dotenv()

# llm = ChatOllama(
#     model="gemma4:31b-cloud", temperature=0.7, api_key=os.getenv("OLLAMA_API_KEY")
# )
llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.7, api_key=os.getenv("GROQ_API")
)
search_tool = get_search_tool()
db = PineconeManager()


def search_node(state: dict):
    print("[SEARCH] Running web search...")
    results = search_tool.invoke({"query": state["query"]})
    return {"search_results": results}


def store_node(state: dict):
    print("[STORE] Saving results to pinecone...")
    for result in state["search_results"]:
        db.upsert_document(
            text=result.get("content", ""),
            metadata={
                "url": result.get("url"),
                "title": result.get("title", ""),
                "content": result.get("content", ""),
            },
        )
    return {"stored_docs": [r.get("title") for r in state["search_results"]]}


def summarize_node(state: dict):
    print("[SUMMARIZE] Generating summary...")
    context = db.query_documents(state["query"])
    context_text = "\n\n".join([doc.get("content", "") for doc in context])

    prompt = ChatPromptTemplate.from_template("""
    You are a research assistant. Answer the user's query using only the provided context.

    Context:
    {context}

    Query:
    {query}

    Answer:
    """)

    active_query = state["refined_query"] or state["query"]
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"context": context_text, "query": active_query})
    return {"summary": summary}


def validator(state: str):
    print("[VALIDATOR] Checking summary...")

    prompt = ChatPromptTemplate.from_template("""
    You are a research critic. Your job is to verify if the summary is relevant to the query.
    Be lenient — if the summary contains ANY relevant information about the query, mark it PASS.

    Query: {query}
    Summary: {summary}

    PASS if:
    - Summary contains any relevant facts related to the query
    - Summary attempts to answer the query even partially

    FAIL only if:
    - Summary is completely unrelated to the query
    - Summary is empty or gibberish

    Respond in EXACTLY this format with no extra text:
    VERDICT: PASS or FAIL
    REFINED_QUERY: (if FAIL, write a better search query to get more relevant results, else write NONE)
    REASON: (one sentence)
    """)

    active_query = state["refined_query"] or state["query"]
    chain = prompt | llm | StrOutputParser()
    result = str(chain.invoke({"query": active_query, "summary": state["summary"]}))

    verdict = "PASS" if "VERDICT: PASS" in result else "FAIL"

    refined_query = active_query

    for line in result.splitlines():
        if line.startswith("REFINED_QUERY") and "NONE" not in line:
            refined_query = line.replace("REFINED_QUERY:", "").strip()
            break

    print(f"[VALIDATOR] Verdict: {verdict}")
    print(f"[VALIDATOR] Refined query: {refined_query}")

    return {
        "validator_result": verdict,
        "refined_query": refined_query,
        "retry_count": state["retry_count"],
    }


MAX_RETRIES = 1


def should_try(state: dict):
    if state["validator_result"] == "FAIL" and state["retry_count"] < MAX_RETRIES:
        print(
            f"[GRAPH] Looping back with refined query (attempt {state['retry_count'] + 1})"
        )
        return "retry"
    return "end"


def retry_node(state: dict):
    return {"query": state["refined_query"], "retry_count": state["retry_count"] + 1}
