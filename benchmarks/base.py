## benchmarks/base.py

from typing import List, Dict, Any
from inspect_ai import Task

class Benchmark:
    name: str
    description: str
    default_split: str
    possible_args: Dict[str, Any]

    @classmethod
    def get_available_splits(cls) -> List[str]:
        """Return a list of available splits for this benchmark."""
        raise NotImplementedError

    @classmethod
    def get_available_subtasks(cls) -> List[str]:
        """Return a list of available subtasks for this benchmark."""
        raise NotImplementedError

    @classmethod
    def validate_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        validated_args = args.copy()  # Start with a copy of all input args
        available_subtasks = cls.get_available_subtasks()
        available_splits = cls.get_available_splits()

        # Handle subtasks
        if 'subtasks' not in validated_args or validated_args['subtasks'] == "all" or not validated_args['subtasks']:
            validated_args['subtasks'] = available_subtasks
        else:
            invalid_subtasks = set(validated_args['subtasks']) - set(available_subtasks)
            if invalid_subtasks:
                raise ValueError(f"Invalid subtasks: {invalid_subtasks}. Available subtasks are: {available_subtasks}")

        # Handle split
        if 'split' in validated_args:
            if validated_args['split'] not in available_splits:
                raise ValueError(f"Invalid split: {validated_args['split']}. Available splits are: {', '.join(available_splits)}")
        else:
            validated_args['split'] = cls.default_split

        # Handle other arguments
        for arg, value in validated_args.items():
            if arg in cls.possible_args:
                if isinstance(cls.possible_args[arg], type):
                    if not isinstance(value, cls.possible_args[arg]):
                        raise ValueError(f"Invalid type for {arg}. Expected {cls.possible_args[arg]}, got {type(value)}")
                elif value not in cls.possible_args[arg]:
                    raise ValueError(f"Invalid value for {arg}. Possible values are {cls.possible_args[arg]}")
            elif arg not in ['subtasks', 'split']:
                raise ValueError(f"Unknown argument: {arg}")

        return validated_args

    @classmethod
    def run(cls, **kwargs) -> Task:
        validated_args = cls.validate_args(kwargs)
        # Implementation using validated_args
        raise NotImplementedError("Subclasses must implement this method")
