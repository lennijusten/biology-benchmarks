#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import argparse

def collect_result_csvs(logs_dir: Path) -> pd.DataFrame:
    """Collect all results.csv files and combine them into a master dataframe."""
    
    dfs = []
    
    for prompt_schema in logs_dir.iterdir():
        if not prompt_schema.is_dir():
            continue
            
        for benchmark in prompt_schema.iterdir():
            if not benchmark.is_dir():
                continue
                
            for model_dir in benchmark.iterdir():
                results_file = model_dir / "results.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        df['benchmark_name'] = benchmark.name
                        df['prompt_schema'] = prompt_schema.name
                        df['model_dir'] = model_dir.name
                        dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
    
    if not dfs:
        raise ValueError("No results.csv files found")
        
    return pd.concat(dfs, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description='Generate master CSV from all experiment results')
    parser.add_argument('logs_dir', type=Path, help='Path to logs directory')
    parser.add_argument('--output', type=Path, default=None, 
                       help='Output path for master CSV (default: logs_dir/master_results.csv)')
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.logs_dir / 'master_results.csv'
        
    master_df = collect_result_csvs(args.logs_dir)
    master_df.to_csv(args.output, index=False)
    print(f"Created master CSV at {args.output}")
    print(f"Total rows: {len(master_df)}")

if __name__ == '__main__':
    main()