# rag/tavily_rag.py

from .base import BaseRAG
from tavily import TavilyClient
import os
import json
from openai import AsyncOpenAI


class TavilyRAG(BaseRAG):
    # TODO: save search query and RAG context

    def __init__(self, **kwargs):
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.kwargs = kwargs

    async def optimize_query(self, query: str, choices_str: str) -> str:
        system_prompt = """You are an expert in crafting optimized search queries. Your task is to take the user input and:
        1) Create a concise, focused Google search query based on the given question and answer choices. This query should be specifically designed to return information that is most useful and relevant for answering the original question. Focus on rare concepts and any specific techniques or problems mentioned.
        
        Provide your response in the following JSON format:
        {
        "google_query": "Your optimized Google search query here",
        }
        """

        full_query = f"{query}\nChoices: {choices_str}"
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_query}
            ],
            response_format={"type": "json_object"}
        )

        optimized_query = json.loads(response.choices[0].message.content)
        return optimized_query['google_query']

    async def retrieve(self, query: str, choices) -> str:
        choices_str = ", ".join(choice.value for choice in choices)
        optimized_query = await self.optimize_query(query, choices_str)

        response = self.tavily_client.search(optimized_query, **self.kwargs)
        context = "\n\n".join(["Source: {} ({})\nContent: {}".format(result['title'], result['url'], result['content']) for result in response['results']])
        return f"Context from internet search:\n{context}"
    