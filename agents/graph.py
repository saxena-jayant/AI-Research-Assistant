from langgraph.graph import StateGraph, END
from .state import ResearchState
from .research_agent import (
    search_node,
    store_node,
    summarize_node,
    validator,
    retry_node,
    should_try,
)


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("search", search_node)
    workflow.add_node("store", store_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("validator", validator)
    workflow.add_node("retry", retry_node)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "store")
    workflow.add_edge("store", "summarize")
    workflow.add_edge("summarize", "validator")
    # workflow.add_edge("validator", END)
    workflow.add_conditional_edges(
        "validator", should_try, {"retry": "retry", "end": END}
    )
    workflow.add_edge("retry", "search")

    return workflow.compile()
