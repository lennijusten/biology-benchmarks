# rag/google.py

from .base import BaseRAG
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import json
from openai import AsyncOpenAI

class GoogleRAG(BaseRAG):
    def __init__(self, **kwargs):
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.kwargs = kwargs

    async def optimize_query(self, query: str, choices_str: str) -> str:
        system_prompt = """You are an expert in crafting optimized search queries. 
        Your task is to take the user input and create a concise, focused Google search query based on the given question and answer choices. 
        This query should be specifically designed to return information that is most useful and relevant for answering the original question. 
        Focus on rare concepts and any specific techniques or problems mentioned.
        
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

    async def google_search(self, query: str, num_results: int = 10):
        service = build("customsearch", "v1", developerKey=self.api_key)
        try:
            res = service.cse().list(q=query, cx=self.cse_id, num=num_results).execute()
            return [{"title": item.get('title', ''), "url": item.get('link', ''), "snippet": item.get('snippet', '')} for item in res.get('items', [])]
        except HttpError as e:
            print(f"An error occurred: {e}")
            return []

    async def retrieve(self, query: str, choices) -> str:
        choices_str = ", ".join(choice.value for choice in choices)
        optimized_query = await self.optimize_query(query, choices_str)

        results = await self.google_search(optimized_query, num_results=self.kwargs.get('num_results', 5))
        context = "\n\n".join([f"Source: {result['title']} ({result['url']})\nContent: {result['snippet']}" for result in results])
        return f"Context from Google search:\n{context}"