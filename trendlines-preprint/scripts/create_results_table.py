import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def format_accuracy(mean, std):
    """Format accuracy as 'mean ± std' with 1 decimal place, converting from proportion to percentage"""
    if pd.isna(mean) or pd.isna(std):
        return ""
    # Multiply by 100 to convert from proportion to percentage
    return f"{(mean * 100):.1f} ± {(std * 100):.1f}"

def transform_benchmark_data(input_file, output_file):
    """
    Transform benchmark data CSV into a table format suitable for a manuscript.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path where the output CSV will be saved
    """
    print(f"Reading data from {input_file}...")
    
    # Read the input CSV
    df = pd.read_csv(input_file, parse_dates=['epoch_model_publication_date'])
    
    # Filter to only include zero_shot prompting schema - create a copy to avoid the SettingWithCopyWarning
    df_zero_shot = df[df['prompt_schema'] == 'zero_shot'].copy()
    
    # Define benchmark mapping (input CSV names to output column names)
    benchmark_mapping = {
        'vct': 'VCT',
        'mmlu': 'MMLU',
        'gpqa': 'GPQA',
        'lab-bench-cloningscenarios': 'CloningScenarios',
        'lab-bench-protocolqa': 'ProtocolQA',
        'lab-bench-litqa2': 'LitQA2',
        'wmdp': 'WMDP',
        'pubmedqa': 'PubMedQA'
    }
    
    # Map benchmark names in the input CSV
    df_zero_shot['benchmark_display'] = df_zero_shot['benchmark'].map(benchmark_mapping)
    
    # List of all benchmarks we want in the final output
    benchmarks = ['PubMedQA', 'MMLU', 'GPQA', 'WMDP', 'LitQA2', 'CloningScenarios', 'ProtocolQA', 'VCT']
    
    # Create pivot tables for mean and std values using the mapped benchmark names
    mean_pivot = pd.pivot_table(
        df_zero_shot,
        values='mean_accuracy',
        index=['inspect_model_name', 'epoch_model_name', 'epoch_organization', 'epoch_model_publication_date'],
        columns='benchmark_display',  # Use the mapped benchmark names
        aggfunc='first'
    ).reset_index()
    
    std_pivot = pd.pivot_table(
        df_zero_shot,
        values='std_accuracy',
        index=['inspect_model_name', 'epoch_model_name', 'epoch_organization', 'epoch_model_publication_date'],
        columns='benchmark_display',  # Use the mapped benchmark names
        aggfunc='first'
    ).reset_index()
    
    # Rename the benchmark columns in the std pivot to avoid column name conflicts when merging
    std_cols_rename = {col: f"{col}_std" for col in std_pivot.columns if col in benchmarks}
    std_pivot.rename(columns=std_cols_rename, inplace=True)
    
    # Also rename the benchmark columns in the mean pivot
    mean_cols_rename = {col: f"{col}_mean" for col in mean_pivot.columns if col in benchmarks}
    mean_pivot.rename(columns=mean_cols_rename, inplace=True)
    
    # Merge the two pivot tables
    result = pd.merge(
        mean_pivot,
        std_pivot,
        on=['inspect_model_name', 'epoch_model_name', 'epoch_organization', 'epoch_model_publication_date']
    )
    
    # Create the formatted accuracy columns
    for benchmark in benchmarks:
        mean_col = f"{benchmark}_mean"
        std_col = f"{benchmark}_std"
        
        if mean_col in result.columns and std_col in result.columns:
            result[benchmark] = result.apply(
                lambda row: format_accuracy(row[mean_col], row[std_col]), 
                axis=1
            )
        else:
            # If the benchmark isn't in the data, add an empty column
            result[benchmark] = ""
    
    # Keep only the columns we need for the final output
    output_columns = ['inspect_model_name', 'epoch_model_name', 'epoch_organization', 'epoch_model_publication_date'] + benchmarks
    result = result[output_columns]
    
    # Sort by organization alphabetically and by publication date in descending order (newer models first)
    result = result.sort_values(['epoch_organization', 'epoch_model_publication_date'], 
                               ascending=[True, False])
    
    # Save the output
    result.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform benchmark data into a table format for manuscript")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", default="model_benchmark_accuracy_table.csv", 
                        help="Path where the output CSV will be saved")
    
    args = parser.parse_args()
    
    transform_benchmark_data(args.input_file, args.output)