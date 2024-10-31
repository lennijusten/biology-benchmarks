# benchmarks/wmdp.py

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from solvers.fewshot_solver import fewshot_solver
from utils.prompts import (
    MULTIPLE_CHOICE_TEMPLATE_COT,
    SINGLE_ANSWER_TEMPLATE,
    FEWSHOT_EXAMPLE_TEMPLATE,
    MULTIPLE_CHOICE_TEMPLATE_FEWSHOT
)
from typing import List, Dict, Any, Optional
import random

WMDP_SPLITS = ["test"]
WMDP_SUBSETS = ["wmdp-bio", "wmdp-cyber", "wmdp-chem"]

def record_to_sample(record: Dict[str, Any]) -> Sample:
    choices = record['choices']
    correct_index = record['answer']
    correct_letter = chr(ord('A') + correct_index)
    
    return Sample(
        input=record['question'],
        target=correct_letter,
        choices=choices
    )

def sample_to_fewshot(sample: Sample, template: str = FEWSHOT_EXAMPLE_TEMPLATE) -> str:
    choices_str = "\n\n".join([f"{chr(ord('A') + i)}) {choice}" for i, choice in enumerate(sample.choices)])
    return template.format(question=sample.input, choices=choices_str, target=sample.target)

@task
def wmdp(subset: str = "wmdp-bio", 
         split: str = "test",
         samples: Optional[int] = None,
         cot: bool = False,
         n_shot: int = 0) -> Task:
    
    if subset not in WMDP_SUBSETS:
        raise ValueError(f"Invalid subset: {subset}. Available subsets are: {WMDP_SUBSETS}")
    
    if split not in WMDP_SPLITS:
        raise ValueError(f"Invalid split: {split}. Available splits are: {WMDP_SPLITS}")
    
    dataset = hf_dataset(
        path="cais/wmdp",
        name=subset,
        split=split,
        sample_fields=record_to_sample,
    )
    
    # Sample if needed
    if samples and samples < len(dataset):
        all_samples = list(dataset)
        random.seed(42)
        sampled_data = random.sample(all_samples, samples)
        dataset = MemoryDataset(sampled_data)
    
    plan = []
    if cot:
        plan.append(multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE_COT))
    elif n_shot > 0:
        if n_shot > len(dataset) - 1:
            raise ValueError(f"n_shot ({n_shot}) must be less than the number of samples in the dataset ({len(dataset)})")
        
        # Get all sample IDs and questions
        all_samples = list(dataset)
        
        # Function to get few-shot examples for a sample
        def get_fewshot_examples(sample_input: str) -> str:
            other_samples = [s for s in all_samples if s.input != sample_input]
            selected_samples = random.sample(other_samples, min(n_shot, len(other_samples)))
            return "\n\n".join([sample_to_fewshot(s) for s in selected_samples])

        plan.append(fewshot_solver(get_fewshot_examples, fewshot_template=MULTIPLE_CHOICE_TEMPLATE_FEWSHOT))
        plan.append(multiple_choice(template=SINGLE_ANSWER_TEMPLATE))
    else:
        plan.append(multiple_choice(template=SINGLE_ANSWER_TEMPLATE))

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=choice(),
    )
    