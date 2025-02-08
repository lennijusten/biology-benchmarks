# benchmarks/vct.py

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from solvers.fewshot_solver import fewshot_solver
from utils.prompts import (
    MULTIPLE_CHOICE_TEMPLATE_COT,
    SINGLE_ANSWER_TEMPLATE,
    FEWSHOT_EXAMPLE_TEMPLATE,
    MULTIPLE_CHOICE_TEMPLATE_FEWSHOT,
    MULTIPLE_ANSWER_TEMPLATE,
    MULTIPLE_ANSWER_TEMPLATE_COT
)
from typing import List, Dict, Any, Optional
import json
import os
import random

VCT_SUBTASKS = ["all", "no_images"]

def load_vct_data(jsonl_path: str, image_folder: str, subtasks: str = "all") -> Dict[str, Any]:
    """
    Load and parse VCT data from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        image_folder: Path to the images folder
        subtasks: Which subset of questions to load ("all" or "no_images")
        
    Returns:
        Dict containing the loaded data
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            question = json.loads(line)
            if subtasks == "no_images" and question["image_file"] is not None:
                continue
            data.append(question)
    return data

def record_to_sample_mc(record: Dict[str, Any], image_folder: str) -> Sample:
    """Convert a VCT record to a multiple-choice Sample."""
    # Build input with statements
    input_text = record["question"].rstrip() + "\n\n"
    for i, stat in enumerate(record["answer_statements"]):
        input_text += f"Statement {i+1}: {stat['statement']}\n"

    # Build choices and find target
    choices = []
    target = None
    for i, choice in enumerate(record["answer_options"]):
        option = " + ".join([f"Statement {x+1}" for x in choice["answer_statement_indices"]])
        if choice["is_correct"]:
            target = chr(ord('A') + i)
        choices.append(option)

    # Create content list for the input
    content = []
    content.append(ContentText(text=input_text))
    if record["image_file"]:
        content.append(ContentImage(image=f"{image_folder}/{record['image_file']}"))

    return Sample(
        input=[ChatMessageUser(content=content if len(content) > 1 else content[0].text)],
        choices=choices,
        target=target,
        metadata={
            "question_id": record["question_id"],
            "image_file": record["image_file"],
            "image_citation": record["image_citation"],
            "expert_approvals": record["expert_approvals"],
            "method": record["method"],
            "explanation": record["explanation"],
            "rubric_elements": record["rubric_elements"],
            "baselining": record["baselining"],
            "canary_string": record["canary_string"]
        }
    )

def record_to_sample_mr(record: Dict[str, Any], image_folder: str) -> Sample:
    """Convert a VCT record to a multiple-response Sample."""
    # Build choices and targets from statements
    choices = []
    targets = []
    for i, statement in enumerate(record["answer_statements"]):
        if statement["is_correct"]:
            targets.append(chr(ord('A') + i))
        choices.append(statement["statement"])

    # Create content list for the input
    content = []
    content.append(ContentText(text=record["question"].rstrip()))
    if record["image_file"]:
        content.append(ContentImage(image=f"{image_folder}/{record['image_file']}"))

    return Sample(
        input=[ChatMessageUser(content=content if len(content) > 1 else content[0].text)],
        choices=choices,
        target=targets,
        metadata={
            "question_id": record["question_id"],
            "image_file": record["image_file"],
            "image_citation": record["image_citation"],
            "expert_approvals": record["expert_approvals"],
            "method": record["method"],
            "explanation": record["explanation"],
            "rubric_elements": record["rubric_elements"],
            "baselining": record["baselining"],
            "canary_string": record["canary_string"]
        }
    )

def sample_to_fewshot(sample: Sample, template: str = FEWSHOT_EXAMPLE_TEMPLATE) -> str:
    """Convert a sample to few-shot example format."""
    choices_str = "\n".join([f"{chr(ord('A') + i)}) {choice}" for i, choice in enumerate(sample.choices)])
    return template.format(
        question=sample.input[0].content if isinstance(sample.input[0].content, str) else sample.input[0].content[0].text,
        choices=choices_str,
        target=sample.target if isinstance(sample.target, str) else ','.join(sample.target)
    )

@task
def vct(mode: str = "mc",
        subtasks: Optional[List[str]] = None,
        samples: Optional[int] = None,
        cot: bool = False,
        n_shot: int = 0) -> Task:
    """
    Run the VCT benchmark.
    
    Args:
        mode: Evaluation mode ("mc" for multiple choice or "mr" for multiple response)
        subtask: Which subset of questions to evaluate ("all" or "no_images")
        samples: Number of samples to use (if None, uses all samples)
        cot: Whether to use chain-of-thought prompting
        n_shot: Number of few-shot examples to use (0 for zero-shot)
    
    Returns:
        Task configured for the specified evaluation
    """
    if mode not in ["mc", "mr"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'mc' or 'mr'")
    
    if subtasks not in VCT_SUBTASKS:
        raise ValueError(f"Invalid subtask: {subtasks}. Available subtasks are: {VCT_SUBTASKS}")
    
    # Load data
    data_path = "vct_data/vct_322Q-shared-set_2025-02-05.jsonl"
    image_folder = "vct_data/vct_data/images"
    
    # Convert records to samples
    record_to_sample = record_to_sample_mc if mode == "mc" else record_to_sample_mr
    data = load_vct_data(data_path, image_folder, subtasks)
    
    all_samples = [record_to_sample(record, image_folder) for record in data]
    
    # Sample if needed
    if samples and samples < len(all_samples):
        random.seed(42)
        sampled_data = random.sample(all_samples, samples)
        dataset = MemoryDataset(sampled_data)
    else:
        dataset = MemoryDataset(all_samples)

    is_multiple_response = mode == "mr"
    
    # Build plan based on configuration
    plan = []
    if cot:
        plan.append(multiple_choice(
            template=MULTIPLE_CHOICE_TEMPLATE_COT,
            multiple_correct=(mode == "mr")
        ))
    elif n_shot > 0:
        if n_shot > len(dataset) - 1:
            raise ValueError(f"n_shot ({n_shot}) must be less than the number of samples in the dataset ({len(dataset)})")
        
        # Function to get few-shot examples for a sample
        def get_fewshot_examples(sample_input: str) -> str:
            other_samples = [s for s in all_samples if s.input != sample_input]
            selected_samples = random.sample(other_samples, min(n_shot, len(other_samples)))
            return "\n\n".join([sample_to_fewshot(s) for s in selected_samples])

        plan.append(fewshot_solver(
            get_fewshot_examples, 
            fewshot_template=MULTIPLE_CHOICE_TEMPLATE_FEWSHOT
        ))
        
        plan.append(multiple_choice(
            template=MULTIPLE_ANSWER_TEMPLATE if is_multiple_response else SINGLE_ANSWER_TEMPLATE,
            multiple_correct=is_multiple_response
        ))

    else:
        plan.append(multiple_choice(
            template=MULTIPLE_ANSWER_TEMPLATE if is_multiple_response else SINGLE_ANSWER_TEMPLATE,
            multiple_correct=is_multiple_response
        ))

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=choice(),
    )