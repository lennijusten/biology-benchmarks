# rag/tools.py

from .tavily import TavilyRAG

RAG_TOOLS = {
    "tavily": TavilyRAG,
    # Add other RAG tools here as they are implemented
}