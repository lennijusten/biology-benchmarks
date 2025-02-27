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

def main(input_csv, output_file):
    df = pd.read_csv(input_csv)

    # Models of interest
    models_of_interest = [
        'anthropic/claude-3-5-sonnet-20241022',
        'google/gemini-1.5-pro',
        'together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
        'openai/gpt-4o'
    ]
    df = df[df['inspect_model_name'].isin(models_of_interest)]
    
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
    
    # Pivot: each row is (benchmark, model), columns for zero_shot, five_shot, etc.
    pivoted = df.pivot_table(
        index=['benchmark', 'inspect_model_name', 'epoch_model_name', 'benchmark_display'],
        columns='prompt_schema',
        values=['mean_accuracy', 'std_accuracy', 'num_runs']
    ).reset_index()
    pivoted.columns = ['_'.join(col).strip('_') for col in pivoted.columns.values]
    
    # Prepare columns for ratio + CI
    pivoted['ratio_five_shot'] = np.nan
    pivoted['ratio_cot'] = np.nan
    pivoted['ci_low_five_shot'] = np.nan
    pivoted['ci_high_five_shot'] = np.nan
    pivoted['ci_low_cot'] = np.nan
    pivoted['ci_high_cot'] = np.nan
    
    # Compute fold differences
    for i, row in pivoted.iterrows():
        # Zero-shot
        mz = row.get('mean_accuracy_zero_shot', np.nan)
        sz = row.get('std_accuracy_zero_shot', np.nan)
        nz = row.get('num_runs_zero_shot', np.nan)
        
        # Five-shot
        mf = row.get('mean_accuracy_five_shot', np.nan)
        sf = row.get('std_accuracy_five_shot', np.nan)
        nf = row.get('num_runs_five_shot', np.nan)
        
        # CoT
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
    
    # Set up the plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")
    
    # 2×2 grid, one subplot per model
    unique_models = pivoted['inspect_model_name'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    strategy_colors = {'five_shot': '#5ab4ac', 'cot': '#d8b365'}
    strategy_labels = {'five_shot': '5-shot', 'cot': 'CoT'}
    
    # Order benchmarks as requested
    benchmark_order = ['pubmedqa', 'mmlu', 'gpqa', 'wmdp', 
                      'lab-bench-litqa2', 'lab-bench-cloningscenarios', 
                      'lab-bench-protocolqa', 'vct']
    benchmark_order_mapping = {b: i for i, b in enumerate(benchmark_order)}
    
    for ax, model in zip(axes, unique_models):
        model_display = pivoted[pivoted['inspect_model_name'] == model]['epoch_model_name'].iloc[0]
        data = pivoted[pivoted['inspect_model_name'] == model].copy()
        
        # Create benchmark ordering
        data['benchmark_order'] = data['benchmark'].map(benchmark_order_mapping)
        data = data.sort_values('benchmark_order')
        
        # y positions for each benchmark
        y_positions = np.arange(len(data))
        
        # Style the subplot
        ax.axvline(1.0, color='#092327', linestyle='--', lw=1.5, alpha=0.7)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Plot each strategy
        for strategy, color, label, offset in [('five_shot', strategy_colors['five_shot'], 
                                               strategy_labels['five_shot'], -0.15),
                                              ('cot', strategy_colors['cot'], 
                                               strategy_labels['cot'], 0.15)]:
            for i, (idx, row) in enumerate(data.iterrows()):
                ratio_col = f'ratio_{strategy}'
                ci_low_col = f'ci_low_{strategy}'
                ci_high_col = f'ci_high_{strategy}'
                
                if pd.notnull(row[ratio_col]):
                    ratio = row[ratio_col]
                    ci_low = row[ci_low_col]
                    ci_high = row[ci_high_col]
                    
                    ax.errorbar(
                        ratio, i + offset,
                        xerr=[[ratio - ci_low], [ci_high - ratio]],
                        fmt='o',
                        color=color,
                        ecolor='gray',
                        capsize=3,
                        markersize=6,
                        alpha=0.8,
                        label=label if i == 0 else ""
                    )
        
        # Improved y-axis labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels([row['benchmark_display'] for _, row in data.iterrows()], 
                          fontsize=11)
        ax.set_ylim(-0.5, len(data) - 0.5)
        
        # Title and axis labels
        ax.set_title(f"{model_display}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Fold difference vs. zero-shot", fontsize=12)
        # Removed y-axis label as requested
        
        # Consistent x-axis limits
        ax.set_xlim(0.7, 1.8)
    
    # Single legend for the entire figure
    handles, labels = [], []
    for strategy, color, label in [('five_shot', strategy_colors['five_shot'], 
                                   strategy_labels['five_shot']),
                                  ('cot', strategy_colors['cot'], 
                                   strategy_labels['cot'])]:
        handles.append(plt.Line2D([0], [0], marker='o', color=color, 
                                 linestyle='None', markersize=8))
        labels.append(label)
    
    fig.legend(handles, labels, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, 0.01), fontsize=12)
    
    plt.suptitle("Effect of Prompting Strategies Relative to Zero-Shot Performance", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot fold differences for five-shot and CoT vs. zero-shot, one subplot per model.")
    parser.add_argument("input_csv", type=str, help="Path to the summary stats CSV file")
    parser.add_argument("--output", type=str, default="fold_plot.png", help="Output image file name")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    main(args.input_csv, out_path)


