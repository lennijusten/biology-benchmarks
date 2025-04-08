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
        parse_dates=['epoch_model_publication_date', 'benchmark_publication_date'],
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

def plot_normalized_benchmark_trends(df: pd.DataFrame, benchmarks: List[str], output_dir: Path):
    """Plot normalized benchmark performance trends over time, similar to the Epoch AI figure."""
    
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
    
    # Initial performance values for older benchmarks (to be replaced with actual values later)
    # These are just placeholders for benchmarks published before your earliest model
    initial_performance_values = {
        'pubmedqa': 0.60,  # Placeholder for initial performance from 2019
        'mmlu': 0.50,      # Placeholder for initial performance from 2021
    }
    
    # Set up the plot style - use consistent color palette
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    sns.set_style("whitegrid")
    
    # Create consistent color mapping based on original order
    original_benchmark_order = [
        'pubmedqa', 'mmlu', 'gpqa', 'wmdp', 'lab-bench-litqa2',
        'lab-bench-cloningscenarios', 'lab-bench-protocolqa', 'vct'
    ]
    color_palette = sns.color_palette("deep", len(original_benchmark_order))
    color_map = {bench: color_palette[i] for i, bench in enumerate(original_benchmark_order)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Filter for zero-shot data
    dfz = df[df['prompt_schema'] == "zero_shot"]
    
    # For x-axis with benchmark publication dates
    benchmark_pub_dates = {}
    all_dates = []
    included_benchmarks = []  # Track which benchmarks are actually included
    
    # For tracking LAB-Bench components (to combine under one label)
    lab_bench_components = ['lab-bench-litqa2', 'lab-bench-cloningscenarios', 'lab-bench-protocolqa']
    
    # Plot data for each benchmark
    for bench in benchmarks:
        df_bench = dfz[dfz['benchmark'] == bench]
        if df_bench.empty:
            continue
            
        # Get expert baseline and benchmark publication date
        baseline_row = df_bench.iloc[0]
        expert_baseline = baseline_row['baselines'].get('expert')
        bench_pub_date = baseline_row['benchmark_publication_date']
        
        # Skip if no expert baseline or publication date available
        if expert_baseline is None or pd.isna(bench_pub_date):
            continue
        
        benchmark_pub_dates[bench] = bench_pub_date
        all_dates.append(bench_pub_date)
        
        # Identify appropriate initial performance
        # For benchmarks released before our earliest model, use placeholder values
        earliest_model_date = df['epoch_model_publication_date'].min()
        
        if bench_pub_date < earliest_model_date and bench in initial_performance_values:
            # Use placeholder for older benchmarks
            initial_performance = initial_performance_values[bench]
            
            # Add the earlier dates to our timeline
            if bench_pub_date not in all_dates:
                all_dates.append(bench_pub_date)
                
        else:
            # For newer benchmarks, use the first model after publication
            relevant_models = df_bench[df_bench['epoch_model_publication_date'] >= bench_pub_date]
            if relevant_models.empty:
                continue
                
            # Get initial model (the one closest to benchmark publication)
            initial_model = relevant_models.nsmallest(1, 'epoch_model_publication_date').iloc[0]
            initial_performance = initial_model['mean_accuracy']
        
        # Skip if initial performance already exceeds expert performance
        if initial_performance >= expert_baseline:
            continue
        
        # Get and sort model publication dates for this benchmark
        model_dates = sorted(df_bench['epoch_model_publication_date'].unique())
        all_dates.extend(model_dates)
        
        # Track performance improvements
        data_points = []
        best_accuracy_so_far = initial_performance
        
        # Always start at -100% at benchmark publication date
        data_points.append({
            'date': bench_pub_date,
            'normalized_accuracy': -100,
            'model': 'Initial',
            'raw_accuracy': initial_performance
        })
        
        # For each date point, only record if there's an improvement
        for date in sorted(model_dates):
            if date < bench_pub_date:
                continue  # Skip models before benchmark
            
            # Get best model performance up to this date
            models_until_date = df_bench[df_bench['epoch_model_publication_date'] <= date]
            if models_until_date.empty:
                continue
            
            best_model = models_until_date.nlargest(1, 'mean_accuracy').iloc[0]
            current_accuracy = best_model['mean_accuracy']
            
            # Only add a point if it's better than what we've seen before
            if current_accuracy > best_accuracy_so_far:
                # Update best seen so far
                best_accuracy_so_far = current_accuracy
                
                # Normalize the performance
                if current_accuracy <= initial_performance:
                    normalized_accuracy = -100  # At or below initial performance
                elif current_accuracy >= expert_baseline:
                    # Calculate percentage above expert baseline using the same scale
                    performance_gap = expert_baseline - initial_performance
                    normalized_accuracy = ((current_accuracy - expert_baseline) / performance_gap) * 100
                else:
                    # Linear scale between -100% and 0%
                    normalized_accuracy = -100 * (expert_baseline - current_accuracy) / (expert_baseline - initial_performance)
                
                data_points.append({
                    'date': date,
                    'normalized_accuracy': normalized_accuracy,
                    'model': best_model['epoch_model_name'],
                    'raw_accuracy': current_accuracy
                })
        
        # Create a DataFrame from the data points
        bench_data = pd.DataFrame(data_points)
        
        if bench_data.empty:
            continue
            
        # Track included benchmarks
        included_benchmarks.append(bench)
            
        # Plot using a regular line instead of step
        line = ax.plot(bench_data['date'], 
                     bench_data['normalized_accuracy'], 
                     label=benchmark_names.get(bench, bench),
                     color=color_map[bench],
                     linewidth=2.5,
                     alpha=0.7)
        
        # Add dots at data points
        ax.scatter(bench_data['date'],
                 bench_data['normalized_accuracy'],
                 color=color_map[bench],
                 s=70,
                 zorder=5,
                 alpha=0.8)
    
    # Add horizontal line at human performance level
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Human expert performance')
    
    # Add horizontal line at initial performance level
    ax.axhline(y=-100, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Initial model performance')
    
    # Add benchmark publication dates as text labels under x-axis
    # Group LAB-Bench components under one label
    lab_bench_date = None
    for bench in included_benchmarks:
        if bench in lab_bench_components:
            if lab_bench_date is None:
                lab_bench_date = benchmark_pub_dates[bench]
                # Add a single "LAB-Bench" label for all components
                ax.text(lab_bench_date, ax.get_ylim()[0] - 5, 
                        "LAB-Bench", 
                        ha='center', va='top', 
                        fontsize=10, rotation=0)
        else:
            # Add individual benchmark labels
            ax.text(benchmark_pub_dates[bench], ax.get_ylim()[0] - 5, 
                    benchmark_names.get(bench, bench), 
                    ha='center', va='top', 
                    fontsize=10, rotation=0)
    
    # Customize plot
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Performance (normalized %)", fontsize=14)
    ax.set_title("LLM Performance on Biology Benchmarks Relative to Human Experts", fontsize=16, pad=20)
    
    # Set y-axis limits with more room for GPQA
    ax.set_ylim(-110, 60)  # Increased upper limit to 60 for GPQA
    
    # Format x-axis with a reasonable time range
    all_dates = sorted(all_dates)
    if all_dates:
        start_date = all_dates[0] - pd.Timedelta(days=365)  # One year before earliest date
        end_date = all_dates[-1] + pd.Timedelta(days=90)    # Three months after latest date
        ax.set_xlim(start_date, end_date)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=4))
    
    # Add grid and customize appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    # Move reference lines to end
    horizontal_line_indices = [i for i, label in enumerate(labels) if 'performance' in label]
    other_indices = [i for i, label in enumerate(labels) if 'performance' not in label]
    
    new_order = other_indices + horizontal_line_indices
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    
    legend = ax.legend(handles, labels,
                      title="Benchmarks and Reference Lines", 
                      loc='center left', 
                      bbox_to_anchor=(1, 0.5),
                      fontsize=12)
    legend.get_frame().set_alpha(1)
    plt.setp(legend.get_title(), fontsize=14, fontweight='bold')
    
    # Add annotation explaining the normalization
    plt.figtext(0.01, 0.01, 
                "Performance normalized with human expert level at 0% and initial model performance at -100%.",
                fontsize=10, alpha=0.7)
    
    # Save the plot
    plt.savefig(output_dir / 'normalized_benchmark_trends.png', 
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
    
    # Generate normalized plot
    plot_normalized_benchmark_trends(df, benchmarks, args.output)

if __name__ == '__main__':
    main()