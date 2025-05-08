import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List

def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess data for plotting."""
    df = pd.read_csv(
        path,
        parse_dates=['epoch_model_publication_date'],
        converters={'baselines': eval}
    )
    
    # Update organization names if needed
    df.loc[df['epoch_organization'] == 'Google DeepMind,Google', 'epoch_organization'] = 'Google DeepMind'
    
    return df.dropna(subset=['epoch_organization'])

def get_color_map(benchmarks: list[str]) -> dict[str, str]:
    palette = sns.color_palette("Set2", len(benchmarks))
    return dict(zip(benchmarks, palette))

def normalize_performance(accuracy: float, expert_baseline: float) -> float:
    """
    Normalize model performance relative to expert baseline.
    
    Returns the difference between model accuracy and expert baseline.
    Positive values indicate performance above expert level.
    
    Args:
        accuracy: Model accuracy (0-1)
        expert_baseline: Expert baseline accuracy (0-1)
        
    Returns:
        float: Normalized performance relative to expert baseline
    """
    # Relative improvement - how much of the remaining "room for improvement" is achieved
    return (accuracy - expert_baseline) / (1 - expert_baseline)
    
    # Alternative approaches:
    # Simple difference approach (model accuracy - expert accuracy)
    # return accuracy - expert_baseline
    
    # Ratio approach (how many times better/worse than expert)
    # return (accuracy / expert_baseline) - 1

def plot_normalized_benchmark_trends(df: pd.DataFrame, benchmarks: List[str], color_map: Dict[str, str], output_dir: Path):
    """
    Plot top model performance trends normalized by expert baselines for each benchmark.
    Zero line represents expert-level performance.
    """
    # Define benchmark name mapping
    benchmark_names = {
        'pubmedqa': 'PubMedQA',
        'mmlu': 'MMLU-Bio',
        'gpqa': 'GPQA-Bio',
        'wmdp': 'WMDP-Bio',
        'lab-bench-litqa2': 'LitQA2',
        'lab-bench-cloningscenarios': 'CloningScenarios',
        'lab-bench-protocolqa': 'ProtocolQA',
        'vct': 'VCT-Text'
    }
    
    # Set up the plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    sns.set_style("whitegrid")
    color_map = get_color_map(benchmarks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Filter for zero-shot data
    dfz = df[df['prompt_schema'] == "zero_shot"]
    publication_dates = sorted(dfz['epoch_model_publication_date'].unique())
    
    # Plot data for each benchmark
    for bench in benchmarks:
        df_bench = dfz[dfz['benchmark'] == bench]
        
        # Skip benchmarks without data
        if df_bench.empty:
            continue
            
        # Check for expert baseline
        first_row = df_bench.iloc[0]
        if 'baselines' not in first_row or not first_row['baselines'] or 'expert' not in first_row['baselines']:
            print(f"Skipping {bench} - no expert baseline available")
            continue
            
        expert_baseline = first_row['baselines']['expert']
        if expert_baseline is None:
            print(f"Skipping {bench} - expert baseline is None")
            continue
        
        # Find top performers at each time point
        top_performers = []
        for step in publication_dates:
            models_at_step = df_bench[df_bench['epoch_model_publication_date'] <= step]
            if not models_at_step.empty:
                top_performer = models_at_step.nlargest(1, 'mean_accuracy').iloc[0]
                
                # Calculate normalized performance
                normalized_accuracy = normalize_performance(top_performer['mean_accuracy'], expert_baseline)
                
                # Create a copy with normalized performance
                performer_copy = top_performer.copy()
                performer_copy['normalized_accuracy'] = normalized_accuracy
                performer_copy['expert_baseline'] = expert_baseline
                
                top_performers.append(performer_copy)
        
        if top_performers:
            bench_data = pd.DataFrame(top_performers)
            
            # Plot regular line function
            line = ax.plot(bench_data['epoch_model_publication_date'], 
                         bench_data['normalized_accuracy'],
                         color=color_map[bench], 
                         label=benchmark_names[bench],
                         linewidth=2,
                         alpha=0.8)
            
            # Add dots in same color as line, excluding the artificially added last point
            line_color = line[0].get_color()
            for _, row in bench_data[:-1].iterrows():  # Exclude the last row which was added for line extension
                ax.scatter(row['epoch_model_publication_date'],
                         row['normalized_accuracy'],
                         color=color_map[bench],
                         s=50,
                         zorder=5,
                         alpha=1)
    
    # --- reference lines ---------------------------------
    expert_ref = ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='Expert performance')
    perfect_ref = ax.axhline(y=1, color='grey', linestyle=':', linewidth=1.2, label='Perfection')   
    
    # Customize plot
    ax.set_xlabel("Model Publication Date", fontsize=14)
    ax.set_ylabel("Model performance relative to experts", fontsize=14)
    ax.set_title("Model Performance Relative to Human Expert Baselines", fontsize=16, pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Add grid and customize appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ---------------- legends ------------------------
    # 1) Benchmarks
    bench_handles, bench_labels = ax.get_legend_handles_labels()
    # filter out the two baseline handles we just added
    baseline_labels = ['Expert performance', 'Perfection']
    bench_pairs = [
        (h, l) for h, l in zip(bench_handles, bench_labels)
        if l not in baseline_labels
    ]
    bench_handles, bench_labels = zip(*bench_pairs)

    bench_legend = ax.legend(
        bench_handles, bench_labels,
        title="Benchmarks",        # bold title
        loc='upper left',
        bbox_to_anchor=(1.02, 1.00),       # to the right of the plot
        borderaxespad=0.,
        fontsize=12
    )
    ax.add_artist(bench_legend)            # keep it when we add the next legend

    # 2) Baselines (Expert & Perfection)
    baseline_legend = ax.legend(
        [expert_ref, perfect_ref],
        ['Expert performance', 'Perfection'],
        title="Baselines",         # bold title
        loc='upper left',
        bbox_to_anchor=(1.02, 0.65),       # slightly below the first legend
        borderaxespad=0.,
        fontsize=12
    )
    plt.setp(bench_legend.get_title(), fontsize=14, fontweight='bold')
    plt.setp(baseline_legend.get_title(), fontsize=14, fontweight='bold')
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_dir / 'normalized_benchmark_trends.png', 
                dpi=300, 
                bbox_inches='tight')
    
    # Create a version with focus on recent performance
    ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2025-04-01'))
    plt.savefig(output_dir / 'normalized_benchmark_trends_recent.png', 
                dpi=300, 
                bbox_inches='tight')
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate normalized benchmark performance plots relative to expert baselines.')
    parser.add_argument('input_csv', type=Path, help='Path to input csv with summary statistics')
    parser.add_argument('--output', type=Path, help='Output directory for plot images')
    args = parser.parse_args()
    
    if not args.output:
        args.output = Path.cwd() / f'plots_{pd.Timestamp.now().strftime("%Y%m%d")}'
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(args.input_csv)
    
    # Define benchmarks to plot
    benchmarks = [
        'pubmedqa', 'mmlu', 'gpqa', 'wmdp', 'lab-bench-litqa2',
        'lab-bench-cloningscenarios', 'lab-bench-protocolqa', 'vct'
    ]
    
    # Create color map
    color_map = get_color_map(benchmarks)
    
    # Generate normalized plot
    plot_normalized_benchmark_trends(df, benchmarks, color_map, args.output)

if __name__ == '__main__':
    main()