# solvers/rag_solver.py

from inspect_ai.solver import solver, TaskState, Generate
from rag.base import BaseRAG

@solver
def rag_solver(rag_instance: BaseRAG):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.input_text
        choices = state.choices if hasattr(state, 'choices') else []
        
        rag_context = await rag_instance.retrieve(query, choices)
        
        if state.messages and state.messages[-1].role == "user":
            state.messages[-1].content = f"{rag_context}\n\nQuestion:\n{state.messages[-1].content}"
        else:
            state.messages.append({"role": "user", "content": f"{rag_context}\n\nQuestion:\n{query}"})
        
        return state

    return solve
