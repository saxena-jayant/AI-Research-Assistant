from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from tools.search_tool import get_search_tool
from utils.pinecone_db import PineconeManager
from dotenv import load_dotenv
import os

load_dotenv()

# llm = ChatOllama(model='gemma4:31b-cloud',temperature=0.7,api_key=os.getenv("OLLAMA_API_KEY"))
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

    prompt = ChatPromptTemplate.from_template(
        """
    You are a research assistant. Answer the user's query using only the provided context.

    Context:
    {context}

    Query:
    {query}

    Answer:
    """
    )

    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"context": context_text, "query": state["query"]})
    return {"summary": summary}
