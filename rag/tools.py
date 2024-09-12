# rag/tools.py

from .tavily import TavilyRAG
from .google import GoogleRAG
from .pubmed import PubMedRAG
from .semantic_scholar import SemanticScholarRAG

RAG_TOOLS = {
    "tavily": TavilyRAG,
    "google": GoogleRAG,
    "pubmed": PubMedRAG,
    "semantic_scholar": SemanticScholarRAG,
}