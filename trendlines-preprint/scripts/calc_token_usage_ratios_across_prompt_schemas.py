import pandas as pd
import argparse
from pathlib import Path

def calculate_token_ratios(summary_file, output_dir):
    """
    Calculate token ratios comparing CoT and five-shot to zero-shot baseline.
    
    Args:
        summary_file: Path to summary CSV file
        output_dir: Directory to save results
    """
    # Read the data
    df = pd.read_csv(summary_file)
    
    # Calculate total tokens
    df['total_tokens'] = df['mean_input_tokens'] + df['mean_output_tokens']
    
    # Pivot tables for each token type
    token_types = {
        'input_tokens': 'mean_input_tokens',
        'output_tokens': 'mean_output_tokens',
        'total_tokens': 'total_tokens'
    }
    
    results = []
    
    # Process each benchmark
    for benchmark in df['benchmark'].unique():
        benchmark_df = df[df['benchmark'] == benchmark]
        
        # Process each model
        for model in benchmark_df['inspect_model_name'].unique():
            model_df = benchmark_df[benchmark_df['inspect_model_name'] == model]
            
            # Get zero-shot baseline values
            zero_shot = model_df[model_df['prompt_schema'] == 'zero_shot']
            if zero_shot.empty:
                continue
                
            zero_shot_values = {
                'input_tokens': zero_shot['mean_input_tokens'].iloc[0],
                'output_tokens': zero_shot['mean_output_tokens'].iloc[0],
                'total_tokens': zero_shot['total_tokens'].iloc[0]
            }
            
            # Process each comparison type (CoT and five-shot)
            for comp_type in ['zero_shot_cot', 'five_shot']:
                comp_data = model_df[model_df['prompt_schema'] == comp_type]
                if comp_data.empty:
                    continue
                
                row = {
                    'model': model,
                    'benchmark': benchmark,
                    'comparison': comp_type
                }
                
                # Add values and calculate ratios for each token type
                for token_type, column in token_types.items():
                    comp_value = comp_data[column].iloc[0]
                    baseline_value = zero_shot_values[token_type]
                    
                    row.update({
                        f'zero_shot_{token_type}': baseline_value,
                        f'{comp_type}_{token_type}': comp_value,
                        f'{comp_type}_ratio_{token_type}': comp_value / baseline_value
                    })
                
                results.append(row)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Sort the DataFrame
    results_df = results_df.sort_values(['benchmark', 'model', 'comparison'])
    
    # Save results
    output_file = Path(output_dir) / 'token_ratios.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate token ratios for different prompting strategies')
    parser.add_argument('summary_file', help='Path to summary CSV file')
    parser.add_argument('--output-dir', default='.', help='Directory to save results')
    args = parser.parse_args()
    
    ratios = calculate_token_ratios(args.summary_file, args.output_dir)
    
    # Print summary statistics
    print("\nMean ratios across all models and benchmarks:")
    for comp_type in ['zero_shot_cot', 'five_shot']:
        print(f"\n{comp_type} vs zero-shot:")
        for token_type in ['input_tokens', 'output_tokens', 'total_tokens']:
            ratio_col = f'{comp_type}_ratio_{token_type}'
            if ratio_col in ratios.columns:
                mean_ratio = ratios[ratio_col].mean()
                print(f"{token_type}: {mean_ratio:.2f}x")