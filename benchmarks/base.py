# benchmarks/base.py

from typing import Dict, Any, List
from inspect_ai import Task

class Benchmark:
    name: str
    description: str
    possible_args: Dict[str, List[str]]

    @classmethod
    def validate_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        validated_args = {}
        for arg, value in args.items():
            if arg in cls.possible_args:
                if value in cls.possible_args[arg]:
                    validated_args[arg] = value
                else:
                    raise ValueError(f"Invalid value for {arg}. Possible values are {cls.possible_args[arg]}")
            else:
                raise ValueError(f"Unknown argument: {arg}")
        return validated_args
    
    @classmethod
    def run(cls, **kwargs) -> Task:
        raise NotImplementedError("Subclasses must implement this method")
