import pandas as pd
import numpy as np

def calculate_factor_increases(csv_file):
    """
    Calculate the fold increase from GPT-3.5 Turbo to the best performing model on each benchmark.
    
    Args:
        csv_file: Path to the summary CSV file
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Filter for zero-shot prompting only
    df = df[df['prompt_schema'] == 'zero_shot']
    
    # Define the GPT-3.5 model names to look for
    gpt35_names = ['openai/gpt-3.5-turbo', 'openai/gpt-3.5-turbo-0125']
    
    # Get unique benchmarks
    benchmarks = df['benchmark'].unique()
    
    # Dictionary to store results
    results = {}
    
    for benchmark in benchmarks:
        benchmark_df = df[df['benchmark'] == benchmark]
        
        # Find GPT-3.5 data
        gpt35_data = benchmark_df[benchmark_df['inspect_model_name'].isin(gpt35_names)]
        
        if len(gpt35_data) == 0:
            print(f"Warning: No GPT-3.5 data found for benchmark {benchmark}")
            continue
        
        gpt35_accuracy = gpt35_data['mean_accuracy'].values[0]
        
        # Find top model
        top_model_row = benchmark_df.loc[benchmark_df['mean_accuracy'].idxmax()]
        top_model_name = top_model_row['epoch_model_name']
        top_model_accuracy = top_model_row['mean_accuracy']
        
        # Calculate factor increase
        factor_increase = top_model_accuracy / gpt35_accuracy
        
        results[benchmark] = {
            'gpt35_accuracy': gpt35_accuracy,
            'top_model_name': top_model_name,
            'top_model_accuracy': top_model_accuracy,
            'factor_increase': factor_increase
        }
    
    # Calculate average factor increase
    avg_factor_increase = np.mean([r['factor_increase'] for r in results.values()])
    
    # Print results
    for benchmark, data in results.items():
        print(f"Benchmark: {benchmark}")
        print(f"GPT-3.5 accuracy - {data['gpt35_accuracy']:.4f}")
        print(f"Top model ({data['top_model_name']}) accuracy - {data['top_model_accuracy']:.4f}")
        print(f"Factor increase - {data['factor_increase']:.2f}x")
        print()
    
    print(f"Average factor increase between GPT-3.5 and top models: {avg_factor_increase:.2f}x")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter the path to the summary CSV file: ")
    
    calculate_factor_increases(csv_file)