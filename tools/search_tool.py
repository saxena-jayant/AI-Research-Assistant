from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

load_dotenv()

def get_search_tool():
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced"
    )