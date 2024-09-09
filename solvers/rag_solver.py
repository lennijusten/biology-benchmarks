# solvers/rag_solver.py

from inspect_ai.solver import solver, TaskState
from rag.base_rag import BaseRAG

@solver
def rag_solver(rag_instance: BaseRAG):
    async def solve(state: TaskState) -> TaskState:
        query = state.input_text
        choices = state.choices if hasattr(state, 'choices') else []
        
        rag_context = await rag_instance.retrieve(query, choices)
        
        if state.messages and state.messages[-1].role == "user":
            state.messages[-1].content = f"{rag_context}" + f"\n\nQuestion:\n{state.messages[-1].content}"
        else:
            state.messages.append({"role": "user", "content": rag_context})
        
        return state

    return solve