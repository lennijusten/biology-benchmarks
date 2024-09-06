# benchmarks/base.py

from typing import List, Dict, Any
from inspect_ai import Task

class Benchmark:
    name: str
    description: str
    hf_hub: str
    default_split: str
    default_subset: str
    possible_args: Dict[str, Any]

    splits: List[str] = []
    subsets: List[str] = []
    subtasks: List[str] = []

    @classmethod
    def get_available_splits(cls) -> List[str]:
        return cls.splits

    @classmethod
    def get_available_subsets(cls) -> List[str]:
        return cls.subsets

    @classmethod
    def get_available_subtasks(cls) -> List[str]:
        return cls.subtasks

    @classmethod
    def validate_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        validated_args = args.copy()

        # Validate split
        if 'split' not in validated_args:
            validated_args['split'] = cls.default_split
        elif validated_args['split'] not in cls.get_available_splits():
            raise ValueError(f"Invalid split: {validated_args['split']}. Available splits are: {cls.get_available_splits()}")

        # Validate subset
        if 'subset' not in validated_args:
            validated_args['subset'] = cls.default_subset
        elif validated_args['subset'] == 'all':
            validated_args['subset'] = cls.get_available_subsets()
        else:
            if isinstance(validated_args['subset'], str):
                validated_args['subset'] = [validated_args['subset']]
            elif not isinstance(validated_args['subset'], list):
                raise ValueError(f"Invalid type for subset. Expected str or list, got {type(validated_args['subset'])}")
            
            invalid_subsets = set(validated_args['subset']) - set(cls.get_available_subsets())
            if invalid_subsets:
                raise ValueError(f"Invalid subsets: {invalid_subsets}. Available subsets are: {cls.get_available_subsets()}")

        # Validate subtasks
        if 'subtasks' in validated_args:
            if validated_args['subtasks'] == 'all' or not validated_args['subtasks']:
                validated_args['subtasks'] = cls.get_available_subtasks()
            else:
                if isinstance(validated_args['subtasks'], str):
                    validated_args['subtasks'] = [validated_args['subtasks']]
                invalid_subtasks = set(validated_args['subtasks']) - set(cls.get_available_subtasks())
                if invalid_subtasks:
                    raise ValueError(f"Invalid subtasks: {invalid_subtasks}. Available subtasks are: {cls.get_available_subtasks()}")
        else:
            validated_args['subtasks'] = cls.get_available_subtasks()

        # Handle other arguments
        for arg, value in validated_args.items():
            if arg in cls.possible_args:
                if isinstance(cls.possible_args[arg], type):
                    if not isinstance(value, cls.possible_args[arg]):
                        raise ValueError(f"Invalid type for {arg}. Expected {cls.possible_args[arg]}, got {type(value)}")
                elif isinstance(cls.possible_args[arg], list) and value not in cls.possible_args[arg]:
                    raise ValueError(f"Invalid value for {arg}. Possible values are {cls.possible_args[arg]}")
            elif arg not in ['split', 'subset', 'subtasks', 'rag_config']:
                raise ValueError(f"Unknown argument: {arg}")

        return validated_args

    @classmethod
    def run(cls, **kwargs) -> Task:
        validated_args = cls.validate_args(kwargs)
        # Implementation using validated_args
        raise NotImplementedError("Subclasses must implement this method")
