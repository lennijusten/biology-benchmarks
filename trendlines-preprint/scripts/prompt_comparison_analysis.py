import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def compute_fold_and_ci(mean_strat, std_strat, n_strat,
                        mean_zero, std_zero, n_zero):
    """
    Computes the fold difference (ratio) = (mean_strat / mean_zero)
    and 95% confidence interval using a log-space (delta-method) approximation.
    
    Returns (ratio, ci_low, ci_high).
    """
    if mean_strat <= 0 or mean_zero <= 0:
        return np.nan, np.nan, np.nan
    
    ratio = mean_strat / mean_zero
    log_ratio = np.log(ratio)
    
    var_strat = (std_strat / mean_strat)**2 / n_strat
    var_zero = (std_zero / mean_zero)**2 / n_zero
    se_log_ratio = np.sqrt(var_strat + var_zero)
    
    ci_log_low = log_ratio - 1.96 * se_log_ratio
    ci_log_high = log_ratio + 1.96 * se_log_ratio
    
    ci_low = np.exp(ci_log_low)
    ci_high = np.exp(ci_log_high)
    return ratio, ci_low, ci_high

def main(input_csv, output_file, csv_output=None):
    df = pd.read_csv(input_csv)

    # Create model mapping to handle model name variations
    model_mapping = {
        'anthropic/claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
        'google/gemini-1.5-pro': 'Gemini 1.5 Pro',
        'together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': 'Llama 3.1-405B',
        'openai/gpt-4o': 'GPT-4o',
        'openai/gpt-4o-2024-08-06': 'GPT-4o'  # Map both versions to same name
    }
    
    # Add a standardized model name column
    df['model_std'] = df['inspect_model_name'].map(lambda x: model_mapping.get(x, x))
    
    # Filter by standardized model names
    models_of_interest = ['Claude 3.5 Sonnet', 'Gemini 1.5 Pro', 'Llama 3.1-405B', 'GPT-4o']
    df = df[df['model_std'].isin(models_of_interest)]
    
    # Benchmark display names
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
    df['benchmark_display'] = df['benchmark'].map(benchmark_mapping)
    
    # Pivot table using standardized model name
    pivoted = df.pivot_table(
        index=['benchmark', 'model_std', 'epoch_model_name', 'benchmark_display'],
        columns='prompt_schema',
        values=['mean_accuracy', 'std_accuracy', 'num_runs']
    ).reset_index()
    pivoted.columns = ['_'.join(col).strip('_') for col in pivoted.columns.values]
    
    # Compute fold differences and CI
    pivoted['ratio_five_shot'] = np.nan
    pivoted['ratio_cot'] = np.nan
    pivoted['ci_low_five_shot'] = np.nan
    pivoted['ci_high_five_shot'] = np.nan
    pivoted['ci_low_cot'] = np.nan
    pivoted['ci_high_cot'] = np.nan
    
    for i, row in pivoted.iterrows():
        # Zero-shot values
        mz = row.get('mean_accuracy_zero_shot', np.nan)
        sz = row.get('std_accuracy_zero_shot', np.nan)
        nz = row.get('num_runs_zero_shot', np.nan)
        
        # Five-shot values
        mf = row.get('mean_accuracy_five_shot', np.nan)
        sf = row.get('std_accuracy_five_shot', np.nan)
        nf = row.get('num_runs_five_shot', np.nan)
        
        # CoT values
        mc = row.get('mean_accuracy_zero_shot_cot', np.nan)
        sc = row.get('std_accuracy_zero_shot_cot', np.nan)
        nc = row.get('num_runs_zero_shot_cot', np.nan)
        
        # 5-shot ratio
        if pd.notnull(mf) and pd.notnull(mz) and mz > 0:
            ratio, ci_low, ci_high = compute_fold_and_ci(mf, sf, nf, mz, sz, nz)
            pivoted.loc[i, 'ratio_five_shot'] = ratio
            pivoted.loc[i, 'ci_low_five_shot'] = ci_low
            pivoted.loc[i, 'ci_high_five_shot'] = ci_high
        
        # CoT ratio
        if pd.notnull(mc) and pd.notnull(mz) and mz > 0:
            ratio, ci_low, ci_high = compute_fold_and_ci(mc, sc, nc, mz, sz, nz)
            pivoted.loc[i, 'ratio_cot'] = ratio
            pivoted.loc[i, 'ci_low_cot'] = ci_low
            pivoted.loc[i, 'ci_high_cot'] = ci_high
    
    # Create a consolidated DataFrame for CSV export
    export_data = []
    
    benchmark_order = ['vct', 'lab-bench-protocolqa', 'lab-bench-cloningscenarios', 
                     'lab-bench-litqa2', 'wmdp', 'gpqa', 'mmlu', 'pubmedqa']
    
    for model in models_of_interest:
        model_data = pivoted[pivoted['model_std'] == model]
        
        for bench in benchmark_order:
            bench_data = model_data[model_data['benchmark'] == bench]
            
            if bench_data.empty:
                # Add placeholder row for missing benchmark
                export_data.append({
                    'model': model,
                    'benchmark': bench,
                    'five_shot_ratio': np.nan,
                    'five_shot_ci_low': np.nan,
                    'five_shot_ci_high': np.nan,
                    'cot_ratio': np.nan,
                    'cot_ci_low': np.nan,
                    'cot_ci_high': np.nan,
                    'zero_shot_accuracy': np.nan,
                    'five_shot_accuracy': np.nan,
                    'cot_accuracy': np.nan
                })
            else:
                # Add data row for existing benchmark
                for _, row in bench_data.iterrows():
                    export_data.append({
                        'model': model,
                        'benchmark': bench,
                        'five_shot_ratio': row.get('ratio_five_shot', np.nan),
                        'five_shot_ci_low': row.get('ci_low_five_shot', np.nan),
                        'five_shot_ci_high': row.get('ci_high_five_shot', np.nan),
                        'cot_ratio': row.get('ratio_cot', np.nan),
                        'cot_ci_low': row.get('ci_low_cot', np.nan),
                        'cot_ci_high': row.get('ci_high_cot', np.nan),
                        'zero_shot_accuracy': row.get('mean_accuracy_zero_shot', np.nan),
                        'five_shot_accuracy': row.get('mean_accuracy_five_shot', np.nan),
                        'cot_accuracy': row.get('mean_accuracy_zero_shot_cot', np.nan)
                    })
    
    # Convert to DataFrame and save to CSV if output path is provided
    export_df = pd.DataFrame(export_data)
    if csv_output:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(csv_path, index=False)
        print(f"Saved fold difference data to {csv_path}")
    
    # Set style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    
    # Create a nested gridspec layout
    outer_grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                                hspace=0.4, wspace=0.3)
    
    strategy_colors = {'five_shot': '#5ab4ac', 'cot': '#d8b365'}
    strategy_labels = {'five_shot': '5-shot', 'cot': 'CoT'}
    
    benchmark_order_mapping = {b: i for i, b in enumerate(benchmark_order)}
    
    # Ensure all benchmarks are included
    all_benchmarks = list(benchmark_order)
    all_benchmarks_display = [benchmark_mapping[b] for b in all_benchmarks]
    
    # Unique models
    unique_models = sorted(pivoted['model_std'].unique())
    
    # Grid positions for each model
    model_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for model_idx, (model, (row, col)) in enumerate(zip(unique_models, model_positions)):
        # Get the epoch_model_name (should be consistent across inspect_model variants)
        model_display = pivoted[pivoted['model_std'] == model]['epoch_model_name'].iloc[0]
        data = pivoted[pivoted['model_std'] == model].copy()
        
        # Create a benchmark_present dictionary to check which benchmarks are present
        benchmarks_present = set(data['benchmark'].unique())
        
        # Create a complete dataset including all benchmarks
        complete_data = []
        for bench in benchmark_order:
            bench_data = data[data['benchmark'] == bench]
            
            if bench not in benchmarks_present:
                # Create a placeholder row for missing benchmark
                placeholder = {
                    'benchmark': bench,
                    'model_std': model,
                    'benchmark_display': benchmark_mapping[bench],
                    'benchmark_order': benchmark_order_mapping[bench],
                    'ratio_five_shot': np.nan,
                    'ratio_cot': np.nan,
                    'ci_low_five_shot': np.nan,
                    'ci_high_five_shot': np.nan,
                    'ci_low_cot': np.nan,
                    'ci_high_cot': np.nan
                }
                complete_data.append(placeholder)
            else:
                # Add existing benchmark data
                for _, row_data in bench_data.iterrows():
                    row_dict = row_data.to_dict()
                    row_dict['benchmark_order'] = benchmark_order_mapping[bench]
                    complete_data.append(row_dict)
        
        # Convert to DataFrame and sort
        complete_df = pd.DataFrame(complete_data)
        complete_df = complete_df.sort_values('benchmark_order')
        
        # Create a nested gridspec for this model (2 columns, 1 row)
        inner_grid = outer_grid[row, col].subgridspec(1, 2, wspace=0.05)
        
        # Create axes for the two panels
        ax1 = fig.add_subplot(inner_grid[0, 0])  # 5-shot panel
        ax2 = fig.add_subplot(inner_grid[0, 1])  # CoT panel
        
        y_positions = np.arange(len(complete_df))
        
        # Plot 5-shot data
        for i, (idx, row_data) in enumerate(complete_df.iterrows()):
            if pd.notnull(row_data.get('ratio_five_shot')):
                ratio = row_data['ratio_five_shot']
                ci_low = row_data['ci_low_five_shot']
                ci_high = row_data['ci_high_five_shot']
                
                ax1.errorbar(
                    ratio, i,
                    xerr=[[ratio - ci_low], [ci_high - ratio]],
                    fmt='o',
                    color=strategy_colors['five_shot'],
                    ecolor='gray',
                    capsize=3,
                    markersize=6,
                    alpha=0.8
                )
        
        # Plot CoT data
        for i, (idx, row_data) in enumerate(complete_df.iterrows()):
            if pd.notnull(row_data.get('ratio_cot')):
                ratio = row_data['ratio_cot']
                ci_low = row_data['ci_low_cot']
                ci_high = row_data['ci_high_cot']
                
                ax2.errorbar(
                    ratio, i,
                    xerr=[[ratio - ci_low], [ci_high - ratio]],
                    fmt='o',
                    color=strategy_colors['cot'],
                    ecolor='gray',
                    capsize=3,
                    markersize=6,
                    alpha=0.8
                )
        
        # Style both subplots
        for ax_idx, (ax, strategy) in enumerate(zip([ax1, ax2], ['5-shot', 'CoT'])):
            # Reference line at 1.0
            ax.axvline(1.0, color='#092327', linestyle='--', lw=1.5, alpha=0.7)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Y-axis labels (only for left subplot)
            ax.set_yticks(y_positions)
            if ax_idx == 0:  # Only add y-tick labels to left subplot of each pair
                ax.set_yticklabels([row_data.get('benchmark_display', '') for _, row_data in complete_df.iterrows()], fontsize=11)
            else:  # Hide y-tick labels on right subplot
                ax.set_yticklabels([])
            
            ax.set_ylim(-0.5, len(complete_df) - 0.5)
            
            # X-axis limits
            ax.set_xlim(0.7, 1.8)
            
            # X-axis label - only for bottom row
            if row == 1:  # Only add for bottom row
                ax.set_xlabel("Fold difference vs. zero-shot", fontsize=12)
            
            # Subplot title (strategy name)
            ax.set_title(strategy, fontsize=12, color=strategy_colors['five_shot' if strategy == '5-shot' else 'cot'])
        
        # Position model title correctly above the pair of panels
        # Calculate the center position between the two axes
        left_pos = ax1.get_position().x0
        right_pos = ax2.get_position().x1
        center_pos = (left_pos + right_pos) / 2
        top_pos = ax1.get_position().y1 + 0.02
        
        # Add model title
        fig.text(center_pos, top_pos, model_display, 
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.suptitle("Effect of Prompting Strategies Relative to Zero-Shot Performance", 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    return export_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot fold differences for five-shot and CoT vs. zero-shot, one subplot per model.")
    parser.add_argument("input_csv", type=str, help="Path to the summary stats CSV file")
    parser.add_argument("--output", type=str, default="fold_plot.png", help="Output image file name")
    parser.add_argument("--csv", type=str, default="fold_plot.csv", help="Output CSV file name for the fold difference data")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    main(args.input_csv, args.output, args.csv)


