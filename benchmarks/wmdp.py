# benchmarks/wmdp.py

from .base import Benchmark
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice
from utils.arg_validation import BenchmarkSchema, ArgumentSchema, validate_rag_config
from solvers.rag_solver import rag_solver
from rag.tools import RAG_TOOLS
from datasets import load_dataset
from typing import List, Dict, Any

WMDP_SPLITS = ["test"]
WMDP_SUBSETS = ["wmdp-bio", "wmdp-cyber", "wmdp-chem"]

class WMDPBenchmark(Benchmark):
    name = "WMDP"
    description = "Weapons of Mass Destruction Proxy Benchmark"
    hf_hub = "cais/wmdp"
    schema = BenchmarkSchema(
        splits=WMDP_SPLITS,
        subsets=WMDP_SUBSETS,
        subtasks=[],
        default_split="test",
        default_subset="wmdp-bio",
        additional_args={
            "samples": ArgumentSchema(int),
            "rag_config": ArgumentSchema(dict, validator=validate_rag_config)
        }
    )

    @classmethod
    @task(category="biology")
    def run(cls, **kwargs) -> Task:
        validated_args = cls.validate_args(kwargs)
        all_samples = []

        subsets_to_process = validated_args['subset']
        if isinstance(subsets_to_process, str):
            subsets_to_process = [subsets_to_process]

        for subset in subsets_to_process:
            try:
                ds = load_dataset(cls.hf_hub, subset, split=validated_args["split"])
                samples = cls.process_subset(subset, validated_args, ds)
                all_samples.extend(samples)
            except Exception as e:
                print(f"Error processing subset {subset}: {str(e)}")
                continue

        if not all_samples:
            raise ValueError("No valid samples were generated. Please check your configuration and try again.")

        plan = []
        rag_config = validated_args.get('rag_config', {})
        if rag_config and rag_config.get('enabled'):
            rag_tool = rag_config.get('tool')
            if rag_tool in RAG_TOOLS:
                rag_class = RAG_TOOLS[rag_tool]
                rag_instance = rag_class(**rag_config.get(rag_tool, {}))
                plan.append(rag_solver(rag_instance))
            else:
                print(f"Warning: RAG tool '{rag_tool}' not found. Skipping RAG.")
        plan.append(multiple_choice())

        return Task(
            dataset=MemoryDataset(all_samples),
            plan=plan,
            scorer=choice()
        )

    @classmethod
    def process_subset(cls, subset: str, validated_args: Dict[str, Any], ds) -> List[Sample]:
        if validated_args.get('samples'):
            ds = ds.shuffle(seed=42).select(range(min(validated_args['samples'], len(ds))))
        
        samples = []
        for idx, item in enumerate(ds):
            choices = item['choices']
            correct_index = item['answer']
            correct_letter = chr(ord('A') + correct_index)
            
            sample = Sample(
                id=f"wmdp_{subset}_{idx}",
                input=item['question'],
                target=correct_letter,
                choices=choices,
                metadata={
                    'subset': subset
                }
            )
            samples.append(sample)

        return samples
    