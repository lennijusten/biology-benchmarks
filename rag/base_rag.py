# rag/base_rag.py

from abc import ABC, abstractmethod
from typing import Iterable

class BaseRAG(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    async def retrieve(self, query: str, choices: Iterable) -> str:
        """Retrieve relevant information and return as a string."""
        pass