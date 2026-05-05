from typing import List, Dict, TypedDict, Annotated
from langgraph.graph import add_messages


class ResearchState(TypedDict):
    query: str
    search_results: List[Dict]
    stored_docs: List[str]
    summary: str
    validator_result: str
    refined_query: str
    retry_count: int
    messages: Annotated[list, add_messages]
