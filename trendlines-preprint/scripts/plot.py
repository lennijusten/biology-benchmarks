import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

def load_plot_data(path: str) -> pd.DataFrame:
    """Load and preprocess data for plotting."""
    df = pd.read_csv(
        path,
        parse_dates=['epoch_model_publication_date', 'eval_start_time', 'eval_end_time', 'results_generated_time'],
        converters={
            'task_args': eval,
            'baselines': eval
        }
    )
    return df

def load_plot_config(path: str) -> dict:
    """Load plot configuration from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)

def plot_benchmark(df: pd.DataFrame, benchmark: str, plot_config: dict, output_dir: Path):
    """Plot benchmark results and save to output directory."""
    plot_zero_shot_trendlines(df, benchmark, plot_config, output_dir)
    plot_top_models_by_promt_schema(df, benchmark, output_dir)
    pass

def main():
    parser = argparse.ArgumentParser(description='Generate and save plots on model performance.')
    parser.add_argument('input_csv', type=Path, help='Path to input csv')
    parser.add_argument('--plot_config', type=str, default='./trendlines-preprint/scripts/plot_config.json', help='Path to plot config'),
    parser.add_argument('--output', type=Path, help='Output dir path for plot images')
    args = parser.parse_args()

    if not args.output:
        args.output = Path.cwd() / f'plots_{pd.Timestamp.now().strftime("%Y%m%d")}'
        args.output.mkdir(parents=True, exist_ok=True)

    df = load_plot_data(args.input_csv, args.plot_config)
    plot_config = load_plot_config(args.plot_config)

    for benchmark in ['mmlu', 'gpqa', 'wmdp', 'lab-bench-litqa2', 'lab-bench-cloningscenarios', 'lab-bench-protocolqa', 'pubmedqa']:
        benchmark_df = df[df['benchmark'] == benchmark]
        if benchmark_df.empty:
            continue

        plot_benchmark(benchmark_df, benchmark, plot_config[benchmark], args.output)

if __name__ == '__main__':
    main()