# utils/arg_validation.py

from typing import Dict, Any, List, Union, Callable
from rag.tools import RAG_TOOLS

class ArgumentSchema:
    def __init__(self, type: type, default: Any = None, choices: List[Any] = None, validator: Callable = None):
        self.type = type
        self.default = default
        self.choices = choices
        self.validator = validator

class BenchmarkSchema:
    def __init__(self, 
                 splits: List[str],
                 subsets: List[str],
                 subtasks: List[str],
                 default_split: str,
                 default_subset: Union[str, List[str]],
                 additional_args: Dict[str, ArgumentSchema] = None):
        self.splits = splits
        self.subsets = subsets
        self.subtasks = subtasks
        self.default_split = default_split
        self.default_subset = default_subset
        self.additional_args = additional_args or {}

def validate_args(args: Dict[str, Any], schema: BenchmarkSchema) -> Dict[str, Any]:
    validated = args.copy()

    # Validate split
    if 'split' not in validated:
        validated['split'] = schema.default_split
    elif validated['split'] not in schema.splits:
        raise ValueError(f"Invalid split: {validated['split']}. Available splits are: {schema.splits}")

    # Validate subset
    validated['subset'] = validate_subset(validated.get('subset', schema.default_subset), schema.subsets)

    # Validate subtasks
    validated['subtasks'] = validate_subtasks(validated.get('subtasks', schema.subtasks), schema.subtasks)

    # Validate additional arguments
    for arg, arg_schema in schema.additional_args.items():
        if arg in validated:
            validated[arg] = validate_argument(validated[arg], arg_schema, arg)
        elif arg_schema.default is not None:
            validated[arg] = arg_schema.default

    return validated

def validate_subset(value: Union[str, List[str]], available_subsets: List[str]) -> List[str]:
    if value == 'all':
        return available_subsets
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"Invalid type for subset. Expected str or list, got {type(value)}")
    invalid_subsets = set(value) - set(available_subsets)
    if invalid_subsets:
        raise ValueError(f"Invalid subsets: {invalid_subsets}. Available subsets are: {available_subsets}")
    return value

def validate_subtasks(value: Union[str, List[str]], available_subtasks: List[str]) -> List[str]:
    if value == 'all' or not value:
        return available_subtasks
    if isinstance(value, str):
        value = [value]
    invalid_subtasks = set(value) - set(available_subtasks)
    if invalid_subtasks:
        raise ValueError(f"Invalid subtasks: {invalid_subtasks}. Available subtasks are: {available_subtasks}")
    return value

def validate_argument(value: Any, schema: ArgumentSchema, arg_name: str) -> Any:
    if not isinstance(value, schema.type):
        raise ValueError(f"Invalid type for {arg_name}. Expected {schema.type}, got {type(value)}")
    if schema.choices and value not in schema.choices:
        raise ValueError(f"Invalid value for {arg_name}. Possible values are {schema.choices}")
    if schema.validator:
        value = schema.validator(value)
    return value

def validate_rag_config(rag_config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(rag_config, dict):
        raise ValueError("rag_config must be a dictionary")
    if rag_config.get('enabled'):
        if 'tool' not in rag_config:
            raise ValueError("rag_config must specify a 'tool' when enabled")
        if rag_config['tool'] not in RAG_TOOLS:
            raise ValueError(f"Unsupported RAG tool: {rag_config['tool']}. Available tools are: {list(RAG_TOOLS.keys())}")
    return rag_config