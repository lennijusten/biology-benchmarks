import json
import csv
import re
import argparse
from typing import Dict, Tuple

def clean_cost(value: str) -> float:
    if not value:
        return 0
    # Remove any non-digit characters except for the decimal point
    cleaned = re.sub(r'[^\d.]', '', value)
    return float(cleaned) if cleaned else 0

def load_model_data(tsv_file: str) -> Dict[str, Dict[str, float]]:
    model_data = {}
    with open(tsv_file, 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            inspect_model_name = row['inspect_model_name']
            model_data[inspect_model_name] = {
                'input_cost_per_M_tokens': clean_cost(row['input_cost_per_M_tokens']),
                'output_cost_per_M_tokens': clean_cost(row['output_cost_per_M_tokens']),
                'cost_per_M_tokens': clean_cost(row['cost_per_M_tokens'])
            }
    return model_data

def calculate_cost(model: str, input_tokens: int, output_tokens: int, model_data: Dict[str, Dict[str, float]]) -> Tuple[float, str]:
    if model not in model_data:
        return 0, f"Model {model} not found in model data"

    data = model_data[model]
    
    if data['input_cost_per_M_tokens'] and data['output_cost_per_M_tokens']:
        input_cost = (data['input_cost_per_M_tokens'] / 1_000_000) * input_tokens
        output_cost = (data['output_cost_per_M_tokens'] / 1_000_000) * output_tokens
        total_cost = input_cost + output_cost
        method = "separate input/output costs"
    elif data['cost_per_M_tokens']:
        total_tokens = input_tokens + output_tokens
        total_cost = (data['cost_per_M_tokens'] / 1_000_000) * total_tokens
        method = "combined cost"
    else:
        return 0, "No cost data available for this model"

    return total_cost, method

def compute_cost_from_log(log_file: str, model_data_file: str) -> None:
    model_data = load_model_data(model_data_file)

    with open(log_file, 'r') as file:
        log = json.load(file)

    total_cost = 0
    total_samples = log['results']['total_samples']
    completed_samples = log['results']['completed_samples']

    print(f"Total questions: {total_samples}")
    print(f"Completed questions: {completed_samples}")
    print()

    for model, usage in log['stats']['model_usage'].items():
        input_tokens = usage['input_tokens']
        output_tokens = usage['output_tokens']
        
        cost, method = calculate_cost(model, input_tokens, output_tokens, model_data)
        total_cost += cost

        print(f"Model: {model}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Cost: ${cost:.6f} (using {method})")
        print()

def main():
    parser = argparse.ArgumentParser(description="Compute cost from Inspect log file using model data.")
    parser.add_argument("log_file", help="Path to the Inspect log file (JSON)")
    parser.add_argument("--model_data", default="models/models_data.tsv", help="Path to the model data TSV file")
    args = parser.parse_args()

    compute_cost_from_log(args.log_file, args.model_data)

if __name__ == "__main__":
    main()