# benchmarks/base.py

from typing import Dict, Any
from inspect_ai import Task
from utils.arg_validation import BenchmarkSchema, validate_args

class Benchmark:
    name: str
    description: str
    hf_hub: str
    schema: BenchmarkSchema

    @classmethod
    def validate_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        return validate_args(args, cls.schema)

    @classmethod
    def run(cls, **kwargs) -> Task:
        validated_args = cls.validate_args(kwargs)
        # Implementation using validated_args
        raise NotImplementedError("Subclasses must implement this method")
