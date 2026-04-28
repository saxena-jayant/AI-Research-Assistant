from langgraph.graph import StateGraph,END
from .state import ResearchState
from .research_agent import search_node,store_node,summarize_node

def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("search", search_node)
    workflow.add_node("store", store_node)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("search")
    workflow.add_edge("search","store")
    workflow.add_edge("store","summarize")
    workflow.add_edge("summarize",END)

    return workflow.compile()