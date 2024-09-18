# solvers/fewshot_solver.py

from inspect_ai.solver import solver, TaskState, Generate
from typing import Callable

@solver
def fewshot_solver(get_fewshot_examples: Callable[[str], str], fewshot_template: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sample_id = state.sample_id
        examples = get_fewshot_examples(sample_id)
        
        # Prepare the choices string
        choices_str = "\n".join([f"{chr(65+i)}. {c.value}" for i, c in enumerate(state.choices)])
        
        # Format the full prompt with few-shot examples
        fewshot_prompt = fewshot_template.format(
            examples=examples,
            question=state.input_text,
            choices=choices_str
        )
        
        # Modify the existing user message or create a new one if it doesn't exist
        if state.messages and state.messages[-1].role == "user":
            state.messages[-1].content = fewshot_prompt
        else:
            state.messages.append({"role": "user", "content": fewshot_prompt})
        
        return state

    return solve