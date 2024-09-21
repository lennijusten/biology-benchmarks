# benchmarks/lab_bench.py

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
from typing import Dict, Any, Optional
import random

LAB_BENCH_SPLITS = ["train"]
LAB_BENCH_SUBSETS = ["LitQA2", "CloningScenarios", "ProtocolQA"]

def record_to_sample_litqa_cloning(record: Dict[str, Any]) -> Sample:
    choices = record['distractors'] + [record['ideal']]
    random.shuffle(choices)
    correct_index = choices.index(record['ideal'])
    correct_letter = chr(ord('A') + correct_index)
    
    return Sample(
        id=f"{record.get('subtask', '')}_{record['id']}",
        input=record['question'],
        target=correct_letter,
        choices=choices,
        metadata={
            'key_passage': record.get('key-passage', ''),
            'sources': record.get('sources', [])
        }
    )

def record_to_sample_protocolqa(record: Dict[str, Any]) -> Sample:
    choices = record['distractors'] + [record['ideal']]
    random.shuffle(choices)
    correct_index = choices.index(record['ideal'])
    correct_letter = chr(ord('A') + correct_index)
    
    return Sample(
        id=f"{record.get('subtask', '')}_{record['id']}",
        input=f"Protocol:\n{record['protocol']}\n\nQuestion:\n{record['question']}",
        target=correct_letter,
        choices=choices,
    )

def sample_to_fewshot(sample: Sample, template: str = FEWSHOT_EXAMPLE_TEMPLATE) -> str:
    choices_str = "\n\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(sample.choices)])
    return template.format(question=sample.input, choices=choices_str, target=sample.target)

@task
def lab_bench(subset: str = "all", 
              split: str = "train",
              samples: Optional[int] = None,
              cot: bool = False,
              n_shot: int = 0) -> Task:
    
    if subset != "all" and subset not in LAB_BENCH_SUBSETS:
        raise ValueError(f"Invalid subset: {subset}. Available subsets are: {LAB_BENCH_SUBSETS} or 'all'")
    
    if split not in LAB_BENCH_SPLITS:
        raise ValueError(f"Invalid split: {split}. Available splits are: {LAB_BENCH_SPLITS}")
    
    subsets_to_process = LAB_BENCH_SUBSETS if subset == "all" else [subset]
    
    all_samples = []
    for current_subset in subsets_to_process:
        
        if current_subset in ["LitQA2", "CloningScenarios"]:
            record_to_sample = record_to_sample_litqa_cloning
        elif current_subset == "ProtocolQA":
            record_to_sample = record_to_sample_protocolqa
        
        dataset = hf_dataset(
            path="futurehouse/lab-bench",
            name=current_subset,
            split=split,
            sample_fields=record_to_sample,
        )
        all_samples.extend(dataset)
    
    # Sample if needed
    if samples and samples < len(all_samples):
        random.seed(42)
        all_samples = random.sample(all_samples, samples)
    
    dataset = MemoryDataset(all_samples)
    
    plan = []
    if cot:
        plan.append(multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE_COT))
    elif n_shot > 0:
        if n_shot > len(dataset) - 1:
            raise ValueError(f"n_shot ({n_shot}) must be less than the number of samples in the dataset ({len(dataset)})")
        
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
    