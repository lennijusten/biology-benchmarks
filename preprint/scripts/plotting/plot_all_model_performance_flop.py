import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_data(input_file: str) -> pd.DataFrame:
    """Load and preprocess data for plotting."""
    df = pd.read_csv(
        input_file,
        parse_dates=['epoch_model_publication_date', 'benchmark_publication_date'],
        converters={'baselines': eval}
    )
    
    # Standardize organization names
    df.loc[df['epoch_organization'] == 'Google DeepMind,Google', 'epoch_organization'] = 'Google DeepMind'
    
    # Filter out missing data for compute
    df = df.dropna(subset=['epoch_organization'])
    
    # Filter for zero-shot results only
    df = df[df['prompt_schema'] == 'zero_shot']
    
    # Check if the compute column exists in the correct format
    if 'epoch_training_compute_flop' in df.columns and 'epoch_training_compute_flops' not in df.columns:
        df['epoch_training_compute_flops'] = df['epoch_training_compute_flop']
    
    return df

def create_benchmark_panel(df: pd.DataFrame, ax, benchmark_name: str, benchmark_display_name: str, color_map: dict):
    """Create a single benchmark panel."""
    # Filter for the specific benchmark
    benchmark_df = df[df['benchmark'] == benchmark_name].copy()
    
    if benchmark_df.empty:
        ax.text(0.5, 0.5, f"No data for {benchmark_display_name}", ha='center', va='center')
        ax.set_title(benchmark_display_name)
        return
    
    # Filter for rows that have compute data
    compute_df = benchmark_df.dropna(subset=['epoch_training_compute_flops'])
    
    if compute_df.empty:
        ax.text(0.5, 0.5, f"No compute data for {benchmark_display_name}", ha='center', va='center')
        ax.set_title(benchmark_display_name)
        return
    
    # Get baseline values from the first row (assuming same for all rows)
    baseline_dict = benchmark_df.iloc[0]['baselines'] if not benchmark_df.empty else {}
    baselines = {}
    
    if baseline_dict:
        if 'expert' in baseline_dict and baseline_dict['expert'] is not None:
            baselines['Expert accuracy'] = {
                'value': baseline_dict['expert'],
                'color': '#092327',
                'linestyle': '--',
                'label': 'Expert accuracy'
            }
        if 'non_expert' in baseline_dict and baseline_dict['non_expert'] is not None:
            baselines['Non-expert accuracy'] = {
                'value': baseline_dict['non_expert'],
                'color': '#0b5351',
                'linestyle': '-.',
                'label': 'Non-expert accuracy'
            }
        if 'random' in baseline_dict and baseline_dict['random'] is not None:
            baselines['Random guess'] = {
                'value': baseline_dict['random'],
                'color': '#00a9a5',
                'linestyle': ':',
                'label': 'Random guess'
            }
    
    # Plot each organization's models - iterate through organizations in alphabetical order
    for org in sorted(compute_df['epoch_organization'].unique()):
        org_data = compute_df[compute_df['epoch_organization'] == org].sort_values('epoch_training_compute_flops')
        
        # For rows with the same model name, take the one with the max mean accuracy 
        # (applies to same model w different reasoning settings)
        org_data = org_data.loc[org_data.groupby('inspect_model_name')['mean_accuracy'].idxmax()]
        
        # Plot scatter points
        ax.scatter(
            org_data['epoch_training_compute_flops'], 
            org_data['mean_accuracy'],
            label=org, 
            color=color_map[org], 
            s=60, 
            alpha=0.8
        )

        # Add error bars
        ax.errorbar(
            org_data['epoch_training_compute_flops'], 
            org_data['mean_accuracy'],
            yerr=org_data['std_accuracy'], 
            color=color_map[org], 
            alpha=0.6,
            fmt='none', 
            capsize=3, 
            capthick=1.5, 
            elinewidth=1.5
        )
    
    # Add baseline lines
    for baseline, properties in baselines.items():
        ax.axhline(
            y=properties['value'], 
            color=properties['color'],
            linestyle=properties['linestyle'], 
            alpha=0.7, 
            linewidth=1.5,
            label=properties['label']
        )
    
    # Set title and labels
    ax.set_title(benchmark_display_name, fontsize=16)
    
    # Format x-axis to log scale for compute
    ax.set_xscale('log')
    
    # Create custom tick formatter to use "1e24" format
    def format_func(x, pos):
        if x == 0:
            return '0'
        exp = int(np.floor(np.log10(x)))
        if exp >= 10:  # Use scientific notation for larger values
            return f'1e{exp}'
        else:
            return f'{x:.1g}'
    
    # Apply the custom formatter
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))
    
    # Ensure appropriate tick placement
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    
    # Format y-axis
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=14)
    
    # Add grid and customize appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_multi_panel_plot(df: pd.DataFrame, output_file: str):
    """Create a multi-panel figure with all benchmarks."""
    # Define benchmark names and display names
    benchmarks = {
        'mmlu': 'MMLU-Bio',
        'gpqa': 'GPQA-Bio',
        'wmdp': 'WMDP-Bio',
        'lab-bench-litqa2': 'LitQA2',
        'lab-bench-cloningscenarios': 'CloningScenarios',
        'lab-bench-protocolqa': 'ProtocolQA',
        'pubmedqa': 'PubMedQA',
        'vct': 'VCT-Text'
    }
    
    # Set up the plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    sns.set_style("whitegrid")
    
    # Create color map for organizations - ordered alphabetically
    organizations = sorted(df['epoch_organization'].unique())
    color_palette = sns.color_palette("deep", len(organizations))
    color_map = dict(zip(organizations, color_palette))
    
    # Create figure with grid
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.3)
    
    # Create panels for each benchmark
    for i, (benchmark_name, benchmark_display) in enumerate(benchmarks.items()):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        create_benchmark_panel(df, ax, benchmark_name, benchmark_display, color_map)
        
        # Only add x-axis label to bottom row
        if row == 1:
            ax.set_xlabel('Training Compute (FLOP)', labelpad=10)
        
        # Only add y-axis label to leftmost panels
        if col == 0:
            ax.set_ylabel('Accuracy')
    
    # Create a separate figure for legend creation to ensure all elements are included
    legend_fig = plt.figure(figsize=(1, 1))
    legend_ax = legend_fig.add_subplot(111)
    
    # Add all legend elements
    # Organizations
    for org in sorted(organizations):
        legend_ax.plot([], [], marker='o', linestyle='', color=color_map[org], label=org, markersize=10)
        
    # Baseline lines - explicitly add all types
    legend_ax.plot([], [], color='#092327', linestyle='--', label='Expert accuracy', linewidth=1.5)
    legend_ax.plot([], [], color='#0b5351', linestyle='-.', label='Non-expert accuracy', linewidth=1.5)
    legend_ax.plot([], [], color='#00a9a5', linestyle=':', label='Random guess', linewidth=1.5)
    
    # Get all handles and labels
    handles, labels = legend_ax.get_legend_handles_labels()
    
    # Close the legend figure as we only need the handles and labels
    plt.close(legend_fig)
    
    # Create the legend on the main figure
    fig.legend(
        handles, 
        labels,
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.05),
        ncol=min(5, len(handles)),
        frameon=True,
        fontsize=14
    )
    
    # Add overall title
    fig.suptitle('Model Performance vs. Training Compute Across Biology Benchmarks', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate multi-panel benchmark performance plot by training compute.')
    parser.add_argument('input_file', type=str, help='Path to summary CSV file')
    parser.add_argument('--output', type=str, default='benchmark_performance_by_compute.png', 
                        help='Output image file path')
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_data(args.input_file)
    
    # Create plot
    create_multi_panel_plot(df, args.output)

if __name__ == "__main__":
    main()