# rag/tools.py

from .tavily import TavilyRAG
from .google import GoogleRAG
from .pubmed import PubMedRAG

RAG_TOOLS = {
    "tavily": TavilyRAG,
    "google": GoogleRAG,
    "pubmed": PubMedRAG,
}