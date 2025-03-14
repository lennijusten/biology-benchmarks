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

def get_color_map(df: pd.DataFrame) -> Dict[str, str]:
    """Create a color map for organizations."""
    organizations = sorted(df['epoch_organization'].unique())
    color_palette = sns.color_palette("deep", len(organizations))
    return dict(zip(organizations, color_palette))

def plot_benchmark_trends(df: pd.DataFrame, benchmarks: List[str], color_map: Dict[str, str], output_dir: Path):
    # Define benchmark name mapping
    benchmark_names = {
        'vct': 'VCT',
        'mmlu': 'MMLU',
        'gpqa': 'GPQA',
        'lab-bench-cloningscenarios': 'CloningScenarios',
        'lab-bench-protocolqa': 'ProtocolQA',
        'lab-bench-litqa2': 'LitQA2',
        'wmdp': 'WMDP',
        'pubmedqa': 'PubMedQA'
    }
    """Plot top model performance trends for each benchmark."""
    
    # Set up the plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Filter for zero-shot data
    dfz = df[df['prompt_schema'] == "zero_shot"]
    publication_dates = sorted(dfz['epoch_model_publication_date'].unique())
    
    # Plot data for each benchmark
    for bench in benchmarks:
        df_bench = dfz[dfz['benchmark'] == bench]
        top_performers = []
        
        for step in publication_dates:
            models_at_step = df_bench[df_bench['epoch_model_publication_date'] <= step]
            if not models_at_step.empty:
                top_performer = models_at_step.nlargest(1, 'mean_accuracy').iloc[0]
                top_performers.append(top_performer)
        
        if top_performers:
            # Add final entry with last performance extending to latest date
            last_performer = top_performers[-1].copy()
            last_performer['epoch_model_publication_date'] = publication_dates[-1]
            top_performers.append(last_performer)
            bench_data = pd.DataFrame(top_performers)
            
            # Plot step function
            line = ax.step(bench_data['epoch_model_publication_date'], 
                         bench_data['mean_accuracy'], 
                         label=benchmark_names[bench],
                         where='post',
                         linewidth=2,
                         alpha=0.4)
            
            # Add dots in same color as line, excluding the artificially added last point
            line_color = line[0].get_color()
            for _, row in bench_data[:-1].iterrows():  # Exclude the last row which was added for line extension
                ax.scatter(row['epoch_model_publication_date'],
                         row['mean_accuracy'],
                         color=line_color,
                         s=50,
                         zorder=5,
                         alpha=0.8)
                
                # # Add model name labels
                # ax.annotate(row['epoch_model_name'],
                #           (row['epoch_model_publication_date'], row['mean_accuracy']),
                #           xytext=(5, 5),
                #           textcoords='offset points',
                #           fontsize=8,
                #           alpha=0.8,
                #           rotation=45,
                #           ha='left',
                #           va='bottom')
    
    # Customize plot
    ax.set_xlabel("Model Publication Date", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Top Model Performance Over Time by Benchmark", fontsize=16, pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Add grid and customize appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create legend
    legend = ax.legend(title="Benchmarks", 
                      loc='center left', 
                      bbox_to_anchor=(1, 0.5),
                      fontsize=14)
    legend.get_frame().set_alpha(1)
    plt.setp(legend.get_title(), fontsize=14, fontweight='bold')
    
    # Remove organization legend since we're not using organization colors anymore
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_trends.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate benchmark performance trend plots.')
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
    color_map = get_color_map(df)
    
    # Generate plot
    plot_benchmark_trends(df, benchmarks, color_map, args.output)

if __name__ == '__main__':
    main()