# benchmarks/pubmedqa.py
# partly based on https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/evals/pubmedqa/pubmedqa.py 

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
from typing import Optional
import random

PUBMEDQA_SPLITS = ["test", "train", "validation"]
PUBMEDQA_SUBSETS = [f"pubmed_qa_labeled_fold{i}_bigbio_qa" for i in range(10)] + ["pubmed_qa_artificial_bigbio_qa", "pubmed_qa_unlabeled_bigbio_qa"]
PUBMEDQA_SUBTASKS = ["labeled"]

def record_to_sample(record) -> Sample:
    choices = {
        "yes": "A",
        "no": "B",
        "maybe": "C",
    }
    context = record["context"]
    question = record["question"]
    return Sample(
        input=f"Context: {context}\n\nQuestion: {question}",
        target=choices[record["answer"][0].lower()],  # provided as e.g. ['yes']
        id=record["id"],
        choices=record["choices"],  # always ['yes, 'no', 'maybe']
        metadata={
            "document_id": record["document_id"],
            "type": record["type"],
        }
    )

def sample_to_fewshot(sample: Sample, template: str = FEWSHOT_EXAMPLE_TEMPLATE) -> str:
    choices_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(sample.choices)])
    return template.format(question=sample.input, choices=choices_str, target=sample.target)

@task
def pubmedqa(subset: str = "pubmed_qa_labeled_fold0_bigbio_qa",
             subtasks: Optional[str] = None,
             split: str = "test",
             samples: Optional[int] = None,
             cot: bool = False,
             n_shot: int = 0) -> Task:
    
    if subset is not None and subset != "all" and subset not in PUBMEDQA_SUBSETS:
            raise ValueError(f"Invalid subset: {subset}. Available subsets are: {PUBMEDQA_SUBSETS} or 'all'")
    
    if split not in PUBMEDQA_SPLITS:
        raise ValueError(f"Invalid split: {split}. Available splits are: {PUBMEDQA_SPLITS}")
    
    if subtasks:
        if subtasks == 'labeled':
            subsets_to_process = [f"pubmed_qa_labeled_fold{i}_bigbio_qa" for i in range(10)]
        else:
            raise ValueError(f"Invalid subtasks: {subtasks}. Available subsets are: {PUBMEDQA_SUBSETS}")
    else:
        subsets_to_process = PUBMEDQA_SUBSETS if subset == "all" else [subset]

    all_samples = []
    for current_subset in subsets_to_process:
        dataset = hf_dataset(
            path="bigbio/pubmed_qa",
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