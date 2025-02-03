import argparse
from pathlib import Path
import pandas as pd
import numpy as np
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

    # Update organization name for specific models
    # This currently only applies to Gemini 2.0 Flash and is presumably an update from EpochAI
    # that considers the merger between DeepMind and Google Brain
    df.loc[df['epoch_organization'] == 'Google DeepMind,Google', 'epoch_organization'] = 'Google DeepMind'

    df = df.dropna(subset=['epoch_organization'])
    return df

def load_plot_config(path: str) -> dict:
    """Load plot configuration from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)

def plot_zero_shot_trendlines(df: pd.DataFrame, benchmark: str, plot_config: dict, color_map: dict, ax):
    """Plot zero-shot performance trends by organization."""
    
    # Filter for zero-shot data
    zshot_df = df[df['prompt_schema'] == 'zero_shot']
    # Exclude specific models
    exclude_models = ['anthropic/claude-3-5-sonnet-20240620']

    if benchmark == 'lab-bench-cloningscenarios':
        exclude_models.append('google/gemini-1.0-pro')

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
    
    for org in color_map.keys():
        org_data = model_stats[model_stats['epoch_organization'] == org].sort_values(by='epoch_model_publication_date')
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
    ax.set_ylabel("Accuracy (0-shot, 10 run)", fontsize=14)
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
    org_handles = [handle for handle, label in zip(handles, labels) if label in color_map.keys()]
    org_labels = [label for label in labels if label in color_map.keys()]
    baseline_handles = [handle for handle, label in zip(handles, labels) if label in baselines.keys()]
    baseline_labels = [label for label in labels if label in baselines.keys()]
    
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
    

def plot_top_models_by_promt_schema(df: pd.DataFrame, plot_config: dict, ax):
    """Plot model performance comparison by prompt schema."""
    
    # Filter for selected models and get data for each schema
    top_models = ['anthropic/claude-3-5-sonnet-20241022', 'openai/gpt-4o', 'google/gemini-1.5-pro', 'together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo']
    df_filtered = df[df['inspect_model_name'].isin(top_models)]
    
    zero_shot_df = df_filtered[df_filtered['prompt_schema'] == 'zero_shot']
    five_shot_df = df_filtered[df_filtered['prompt_schema'] == 'five_shot']
    zero_shot_cot_df = df_filtered[df_filtered['prompt_schema'] == 'zero_shot_cot']
    
    # Colors for different conditions
    colors = {
        'Zero-shot': '#5ab4ac',
        'Five-shot': '#f6e8c3',
        'Zero-shot CoT': '#d8b365'
    }
    
    dot_colors = {
        'Zero-shot': '#387b75',
        'Five-shot': '#c4b68c',
        'Zero-shot CoT': '#8e6c24'
    }
    
    # Prepare combined dataframe
    zero_shot_df = zero_shot_df.copy()
    zero_shot_df.loc[:, 'condition'] = 'Zero-shot'
    combined_dfs = [zero_shot_df]

    if not five_shot_df.empty:
        five_shot_df = five_shot_df.copy()
        five_shot_df.loc[:, 'condition'] = 'Five-shot'
        combined_dfs.append(five_shot_df)

    if not zero_shot_cot_df.empty:
        zero_shot_cot_df = zero_shot_cot_df.copy()
        zero_shot_cot_df.loc[:, 'condition'] = 'Zero-shot CoT'
        combined_dfs.append(zero_shot_cot_df)
    
    combined_df = pd.concat(combined_dfs)
    num_conditions = len(combined_dfs)
    
    # Sort models by median zero-shot accuracy
    model_order = zero_shot_df.groupby('epoch_model_name')['accuracy'].median().sort_values(ascending=False).index
    
    # Create box plot
    sns.boxplot(x='accuracy', y='epoch_model_name', hue='condition', 
                data=combined_df, order=model_order,
                palette=colors, width=0.7, showfliers=False, dodge=True)
    
    # Add individual points
    conditions = list(colors.keys())[:num_conditions]
    box_width = 0.7 / num_conditions
    
    for i, condition in enumerate(conditions):
        data = combined_df[combined_df['condition'] == condition]
        color = dot_colors[condition]
        
        y_offset = -0.25 + (0.5 * i / (num_conditions - 1)) if num_conditions > 1 else 0
        
        for j, model in enumerate(model_order):
            model_data = data[data['epoch_model_name'] == model]
            y_positions = np.random.normal(j + y_offset, 0.05, len(model_data))
            ax.scatter(model_data['accuracy'], y_positions,
                      color=color, alpha=0.7, s=15, zorder=10)
    
    # Customize plot
    ax.set_xlabel('Accuracy (10 runs)', fontsize=14)
    ax.set_ylabel('')
    ax.set_xlim(plot_config['xlim'])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add baselines
    baseline_dict = df.iloc[0]['baselines']
    if baseline_dict:
        if 'expert' in baseline_dict and baseline_dict['expert'] is not None:
            ax.axvline(x=baseline_dict['expert'], color='#092327',
                      linestyle='--', alpha=0.7, label='Expert accuracy')
        if 'non_expert' in baseline_dict and baseline_dict['non_expert'] is not None:
            ax.axvline(x=baseline_dict['non_expert'], color='#0b5351',
                      linestyle='-.', alpha=0.7, label='Non-expert accuracy')
        if 'random' in baseline_dict and baseline_dict['random'] is not None:
            ax.axvline(x=baseline_dict['random'], color='#00a9a5',
                      linestyle=':', alpha=0.7, label='Random guess')
    
    # Customize legend
    if num_conditions > 1:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[:num_conditions], labels[:num_conditions],
                         title='Condition', fontsize=10, loc='best')
        legend.get_frame().set_alpha(1)
        plt.setp(legend.get_title(), fontsize=12, fontweight='bold')
    else:
        ax.get_legend().remove()
    
    plt.tight_layout()

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

    plot_zero_shot_trendlines(df, benchmark, plot_config, color_map, ax1)
    plot_top_models_by_promt_schema(df, plot_config, ax2)

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
        print(f"Plotting {benchmark} benchmark...")
        benchmark_df = df[df['benchmark'] == benchmark]
        if benchmark_df.empty:
            continue

        plot_benchmark(benchmark_df, benchmark, plot_config[benchmark], args.output)

if __name__ == '__main__':
    main()