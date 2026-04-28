from typing import List,Dict,TypedDict,Annotated
from langgraph.graph import add_messages

class ResearchState(TypedDict):
    query:str
    search_results:List[Dict]
    stored_docs:List[str]
    summary:str
    messages:Annotated[list,add_messages]