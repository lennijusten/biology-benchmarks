# rag/semantic_scholar.py

import aiohttp
import json
import asyncio
import random
from .base import BaseRAG
from openai import AsyncOpenAI
import os

class SemanticScholarRAG(BaseRAG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.num_results = kwargs.get('num_results', 5)
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 1)  # 1 second delay between requests
        self.max_retries = kwargs.get('max_retries', 5)

    async def optimize_query(self, query: str, choices_str: str) -> str:
        system_prompt = """You are an expert in crafting optimized search queries for the Semantic Scholar API. 
        Create a concise search query based on the given question and answer choices. 
        Focus on the main concepts and use no more than 3-4 key terms.
        
        Provide your response in the following JSON format:
        {
        "search_query": "Your optimized search query here",
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
        print(f"Optimized query: {optimized_query['search_query']}")
        return optimized_query['search_query']

    async def semantic_scholar_search(self, query: str):
        search_url = f"{self.base_url}/paper/search"
        params = {
            'query': query,
            'limit': self.num_results,
            'fields': 'paperId,title,abstract,year,authors.name'
        }

        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit_delay)  # Rate limiting delay
                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=params) as response:
                        if response.status == 429:
                            delay = 2 ** attempt + random.random()  # Exponential backoff with jitter
                            print(f"Rate limited. Retrying in {delay:.2f} seconds...")
                            await asyncio.sleep(delay)
                            continue
                        elif response.status != 200:
                            print(f"Error: {response.status}")
                            return []
                        data = await response.json()
                        return data.get('data', [])
            except aiohttp.ClientError as e:
                print(f"Request failed: {e}")
                if attempt == self.max_retries - 1:
                    return []
                delay = 2 ** attempt + random.random()
                await asyncio.sleep(delay)

        return []  # If we've exhausted all retries

    async def retrieve(self, query: str, choices) -> str:
        choices_str = ", ".join(choice.value for choice in choices)
        optimized_query = await self.optimize_query(query, choices_str)

        results = await self.semantic_scholar_search(optimized_query)
        if not results:
            results = await self.semantic_scholar_search(query)

        context = "\n\n".join([f"Title: {paper['title']}\nAuthors: {', '.join(author['name'] for author in paper['authors'])}\nYear: {paper['year']}\nAbstract: {paper.get('abstract', 'No abstract available')}" for paper in results])
        return f"Context from Semantic Scholar search:\n{context}"