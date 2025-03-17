import pandas as pd
import os
from pathlib import Path
import argparse

def process_tokens_data(input_file, output_dir):
    """Process the input CSV and generate token summary files by prompt schema."""
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of unique prompt schemas
    prompt_schemas = df['prompt_schema'].unique()
    
    # Process each prompt schema
    for schema in prompt_schemas:
        # Filter data for current schema
        schema_df = df[df['prompt_schema'] == schema]
        
        # Initialize result dictionary
        result_data = {'inspect_model_name': []}
        benchmarks = sorted(schema_df['benchmark'].unique())
        
        # Initialize columns for each benchmark
        for benchmark in benchmarks:
            result_data[f'{benchmark}_input_tokens'] = []
            result_data[f'{benchmark}_output_tokens'] = []
        
        # Get unique models
        models = sorted(schema_df['inspect_model_name'].unique())
        
        # Process each model
        for model in models:
            result_data['inspect_model_name'].append(model)
            
            # Get token counts for each benchmark
            for benchmark in benchmarks:
                benchmark_data = schema_df[
                    (schema_df['inspect_model_name'] == model) & 
                    (schema_df['benchmark'] == benchmark)
                ]
                
                # Calculate mean tokens if data exists
                input_tokens = int(benchmark_data['mean_input_tokens'].mean()) if not benchmark_data.empty else ''
                output_tokens = int(benchmark_data['mean_output_tokens'].mean()) if not benchmark_data.empty else ''
                
                result_data[f'{benchmark}_input_tokens'].append(input_tokens)
                result_data[f'{benchmark}_output_tokens'].append(output_tokens)
        
        # Create DataFrame and save to CSV
        result_df = pd.DataFrame(result_data)
        output_file = os.path.join(output_dir, f'token_summary_{schema}.csv')
        result_df.to_csv(output_file, index=False)
        print(f"Generated summary for {schema} at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate token usage summaries by prompt schema')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', type=Path, help='Output directory for token data')
    args = parser.parse_args()
    
    if not args.output:
        args.output = Path.cwd() / f'token_data_{pd.Timestamp.now().strftime("%Y%m%d")}'
    args.output.mkdir(parents=True, exist_ok=True)
    
    process_tokens_data(args.input_file, args.output)