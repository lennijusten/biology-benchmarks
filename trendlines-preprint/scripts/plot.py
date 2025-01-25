import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns

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

def plot_zero_shot_trendlines(df: pd.DataFrame, benchmark: str, plot_config: dict):
    """Plot zero-shot performance trends by organization."""
    
    # Filter for zero-shot data
    zshot_df = df[(df['prompt_schema'] == 'zero_shot') & (df['inspect_model_name'] != 'anthropic/claude-3-5-sonnet-20240620')]
    # Exclude specific models
    exclude_models = ['anthropic/claude-3-5-sonnet-20240620']
    zshot_df = zshot_df[~zshot_df['inspect_model_name'].isin(exclude_models)]
    
    # Calculate mean and std of accuracy for each model
    model_stats = zshot_df.groupby(['epoch_organization', 'inspect_model_name', 'epoch_model_name', 'epoch_model_publication_date'])['accuracy'].agg(['mean', 'std']).reset_index()
    # Add a column with the number of results for each model
    model_stats['num_results'] = zshot_df.groupby(['epoch_organization', 'inspect_model_name', 'epoch_model_name', 'epoch_model_publication_date']).size().values
    # Drop rows where num_results is not equal to 10
    dropped_rows = model_stats[model_stats['num_results'] != 10]
    if not dropped_rows.empty:
        print(f"Dropping {len(dropped_rows)} rows where num_results != 10:")
        for _, row in dropped_rows.iterrows():
            print(f"Organization: {row['epoch_organization']}, Model: {row['epoch_model_name']}, Num results: {row['num_results']}")
    model_stats = model_stats[model_stats['num_results'] == 10]
    
    # Create color map for organizations
    organizations = sorted(model_stats['epoch_organization'].unique())
    color_palette = sns.color_palette("deep", len(organizations))
    color_map = dict(zip(organizations, color_palette))
    
    # Get manual offsets and plot limits
    manual_offsets = plot_config['offsets']
    ylim = plot_config['ylim']
    xlim = plot_config['xlim']
    
    # Get baseline values from first row (assuming same for all rows)
    baseline_dict = df.iloc[0]['baselines']
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
    
    # Plot data
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    for org in color_map.keys():
        org_data = model_stats[model_stats['epoch_organization'] == org]
        if not org_data.empty:
            ax.scatter(org_data['epoch_model_publication_date'], org_data['mean'],
                      label=org, color=color_map[org], s=80, alpha=0.7)
            
            # Add error bars
            ax.errorbar(org_data['epoch_model_publication_date'], org_data['mean'],
                       yerr=org_data['std'], color=color_map[org], alpha=0.4,
                       fmt='none', capsize=3, capthick=1.5, elinewidth=1.5)
            
            # Connect points with lines
            ax.plot(org_data['epoch_model_publication_date'], org_data['mean'],
                   color=color_map[org], alpha=0.4, linewidth=2)
            
            # Add model name annotations
            for _, row in org_data.iterrows():
                offset = manual_offsets.get(row['epoch_model_name'], (5, 5))
                ax.annotate(row['epoch_model_name'],
                          (row['epoch_model_publication_date'], row['mean']),
                          xytext=offset, textcoords='offset points',
                          fontsize=10, alpha=0.8, rotation=0, ha='left', va='bottom')
    
    # Add baseline lines
    for baseline, properties in baselines.items():
        ax.axhline(y=properties['value'], color=properties['color'],
                  linestyle=properties['linestyle'], alpha=0.7, linewidth=2.0,
                  label=properties['label'])
    
    # Customize plot
    ax.set_xlabel("Model publication date", fontsize=14)
    ax.set_ylabel("Accuracy (0-shot, 1 run)", fontsize=14)
    ax.set_ylim(ylim)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Add grid and customize appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create legend with sections
    handles, labels = ax.get_legend_handles_labels()
    org_handles = handles[:len(color_map)]
    org_labels = labels[:len(color_map)]
    baseline_handles = handles[len(color_map):]
    baseline_labels = labels[len(color_map):]
    
    legend_elements = [
        plt.Line2D([0], [0], color='w', alpha=0, label='Organization'),
        *[plt.Line2D([0], [0], color=color_map[org], lw=4, alpha=0.7, label=org)
          for org in org_labels if org != 'nan'],
        plt.Line2D([0], [0], color='w', alpha=0, label=' '),
        plt.Line2D([0], [0], color='w', alpha=0, label='Baselines'),
        *[plt.Line2D([0], [0], color=baselines[label]['color'],
                     linestyle=baselines[label]['linestyle'],
                     lw=2, label=label, alpha=0.7)
          for label in baseline_labels]
    ]
    
    legend = ax.legend(handles=legend_elements, loc='best', fontsize=12)
    legend.get_frame().set_alpha(1)
    
    # Set bold font for section titles
    for text in legend.get_texts():
        if text.get_text() in ['Organization', 'Baselines']:
            text.set_fontweight('bold')
            text.set_fontsize(12)
    
    plt.tight_layout()
    plt.show()
    plt.pause(1)
    pass
    

def plot_top_models_by_promt_schema(df: pd.DataFrame, benchmark: str, plot_config: dict):
    pass

def plot_benchmark(df: pd.DataFrame, benchmark: str, plot_config: dict, output_dir: Path):
    """Plot benchmark results and save to output directory."""

    # Set up the plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300, gridspec_kw={'width_ratios': [1.5, 1]})

    # Create a color map for organizations
    organizations = sorted(df['epoch_organization'].unique())
    color_palette = sns.color_palette("deep", len(organizations))
    color_map = dict(zip(organizations, color_palette))

    plot_zero_shot_trendlines(df, benchmark, plot_config)
    plot_top_models_by_promt_schema(df, benchmark, plot_config)

    fig.suptitle(f"Model Performance on {plot_config['name']} Benchmark", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{plot_config['name'].lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate and save plots on model performance.')
    parser.add_argument('input_csv', type=Path, help='Path to input csv')
    parser.add_argument('--plot_config', type=str, default='./trendlines-preprint/scripts/plot_config.json', help='Path to plot config'),
    parser.add_argument('--output', type=Path, help='Output dir path for plot images')
    args = parser.parse_args()

    if not args.output:
        args.output = Path.cwd() / f'plots_{pd.Timestamp.now().strftime("%Y%m%d")}'
        args.output.mkdir(parents=True, exist_ok=True)

    df = load_plot_data(args.input_csv)
    plot_config = load_plot_config(args.plot_config)

    for benchmark in ['mmlu', 'gpqa', 'wmdp', 'lab-bench-litqa2', 'lab-bench-cloningscenarios', 'lab-bench-protocolqa', 'pubmedqa']:
        benchmark_df = df[df['benchmark'] == benchmark]
        if benchmark_df.empty:
            continue

        plot_benchmark(benchmark_df, benchmark, plot_config[benchmark], args.output)

if __name__ == '__main__':
    main()