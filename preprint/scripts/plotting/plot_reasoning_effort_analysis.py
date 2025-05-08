import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import json
from pathlib import Path
import re

def load_and_prepare_data(csv_file):
    """Load and preprocess the summary CSV file."""
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Convert task_args from string to dictionary
    def parse_task_args(task_args):
        if pd.isna(task_args):
            return {}
        
        # If it's already a dictionary, return it
        if isinstance(task_args, dict):
            return task_args
            
        # For string representation, handle it more robustly
        try:
            # First approach: Use safer eval to parse the dictionary
            if isinstance(task_args, str):
                try:
                    import ast
                    return ast.literal_eval(task_args)
                except (SyntaxError, ValueError):
                    pass
            
            # Second approach: Manual JSON conversion
            # Replace single quotes with double quotes for valid JSON
            task_args_json = task_args.replace("'", '"')
            
            # Handle nested list representations like "['Biology']"
            task_args_json = re.sub(r'"\[(.*?)\]"', r'[\1]', task_args_json)
            
            # Fix potential issues with "None" values
            task_args_json = task_args_json.replace(': None', ': null')
            
            # Fix potential issues with "True" and "False" values
            task_args_json = task_args_json.replace(': True', ': true').replace(': False', ': false')
            
            return json.loads(task_args_json)
        except Exception as e:
            # Instead of printing, just return an empty dict without error message
            # print(f"Warning: Could not parse task args: {task_args} - {str(e)}")
            return {}
    
    df['task_args_dict'] = df['task_args'].apply(parse_task_args)
    
    # Extract reasoning parameters
    df['reasoning_tokens'] = df['task_args_dict'].apply(
        lambda x: x.get('reasoning_tokens', None) if isinstance(x, dict) else None
    )
    
    df['reasoning_effort'] = df['task_args_dict'].apply(
        lambda x: x.get('reasoning_effort', None) if isinstance(x, dict) else None
    )
    
    return df

def get_reasoning_data(df, benchmark_name):
    """Extract data for reasoning models on a specific benchmark."""
    # Filter for the benchmark
    df_benchmark = df[df['benchmark'] == benchmark_name]
    
    # Extract data for Claude 3.7 Sonnet
    claude_data = df_benchmark[df_benchmark['inspect_model_name'] == 'anthropic/claude-3-7-sonnet-20250219'].copy()
    
    # Add a category for "no reasoning" (default)
    claude_no_reasoning = claude_data[claude_data['reasoning_tokens'].isna()].copy()
    if not claude_no_reasoning.empty:
        claude_no_reasoning['reasoning_category'] = "No reasoning limit specified"
    
    # Get rows with reasoning tokens
    claude_with_reasoning = claude_data[claude_data['reasoning_tokens'].notna()].copy()
    if not claude_with_reasoning.empty:
        claude_with_reasoning['reasoning_category'] = claude_with_reasoning['reasoning_tokens'].apply(
            lambda x: f"{int(x/1000)}k reasoning token limit"
        )
    
    # Combine the two
    claude_combined = pd.concat([claude_no_reasoning, claude_with_reasoning])
    
    # Extract data for o3-mini
    o3_data = df_benchmark[df_benchmark['inspect_model_name'] == 'openai/o3-mini-2025-01-31'].copy()
    
    # Add category for default "medium" reasoning
    o3_default = o3_data[o3_data['reasoning_effort'].isna()].copy()
    if not o3_default.empty:
        o3_default['reasoning_category'] = "Medium reasoning effort"
        o3_default['reasoning_effort'] = "medium"
    
    # Get rows with specified reasoning effort
    o3_with_effort = o3_data[o3_data['reasoning_effort'].notna()].copy()
    if not o3_with_effort.empty:
        o3_with_effort['reasoning_category'] = o3_with_effort['reasoning_effort'].apply(
            lambda x: f"{x.capitalize()} reasoning effort"
        )
    
    # Combine the two
    o3_combined = pd.concat([o3_default, o3_with_effort])
    
    # Return both datasets
    return claude_combined, o3_combined

def plot_single_benchmark(ax, claude_data, o3_data, benchmark_name, sample_counts=None):
    """Plot a single benchmark subplot."""
    # Define colors to match other plots
    claude_color = "#5ab4ac"  # Light blue-green
    o3_color = "#d8b365"      # Light brown
    
    # Map from benchmark name to display name
    benchmark_display = {
        "vct": "VCT-Text",
        "gpqa": "GPQA-Bio",
        "lab-bench-cloningscenarios": "CloningScenarios",
        "lab-bench-protocolqa": "ProtocolQA"
    }
    
    display_name = benchmark_display.get(benchmark_name, benchmark_name)
    
    # Define markers for different reasoning settings
    claude_markers = {
        "No reasoning limit specified": "o",
        "4k reasoning token limit": "^", 
        "16k reasoning token limit": "s"
    }
    
    o3_markers = {
        "Low reasoning effort": "o",
        "Medium reasoning effort": "^",
        "High reasoning effort": "s"
    }
    
    # Plot Claude data if available
    if not claude_data.empty:
        # Group by reasoning category and aggregate
        claude_grouped = claude_data.groupby('reasoning_category').agg({
            'mean_output_tokens': 'mean',
            'mean_accuracy': 'mean',
            'std_accuracy': 'mean',  # Using mean of standard deviations
            'total_samples': 'first'  # Get the sample count
        }).reset_index()
        
        # Calculate mean tokens per question by dividing by the number of samples
        claude_grouped['tokens_per_question'] = claude_grouped['mean_output_tokens'] / claude_grouped['total_samples']
        
        # Store sample count if provided
        if sample_counts is not None and not claude_grouped.empty:
            sample_counts[benchmark_name] = int(claude_grouped['total_samples'].iloc[0])
        
        # Sort by token count (no reasoning should be first)
        claude_grouped = claude_grouped.sort_values('tokens_per_question')
        
        # First plot the connecting line
        ax.plot(
            claude_grouped['tokens_per_question'],
            claude_grouped['mean_accuracy'] * 100,
            color=claude_color,
            linewidth=2,
            alpha=0.8
        )
        
        # Then plot each point separately with the appropriate marker
        for _, row in claude_grouped.iterrows():
            category = row['reasoning_category']
            marker = claude_markers.get(category, "o")
            
            ax.errorbar(
                row['tokens_per_question'], 
                row['mean_accuracy'] * 100,  # Convert to percentage
                yerr=row['std_accuracy'] * 100,  # Convert to percentage
                fmt=marker, 
                markersize=8, 
                capsize=3, 
                capthick=1.5,
                elinewidth=1.5,
                color=claude_color,
                markeredgecolor=claude_color,  # Set marker edge color to match fill
                label=f'Claude 3.7 Sonnet ({category})' if _ == 0 else "",
                alpha=0.8
            )
    
    # Plot o3-mini data if available
    if not o3_data.empty:
        # Group by reasoning category and aggregate
        o3_grouped = o3_data.groupby('reasoning_category').agg({
            'mean_output_tokens': 'mean',
            'mean_accuracy': 'mean',
            'std_accuracy': 'mean',  # Using mean of standard deviations
            'total_samples': 'first'  # Get the sample count
        }).reset_index()
        
        # Calculate mean tokens per question by dividing by the number of samples
        o3_grouped['tokens_per_question'] = o3_grouped['mean_output_tokens'] / o3_grouped['total_samples']
        
        # Store sample count if provided
        if sample_counts is not None and not o3_grouped.empty:
            sample_counts[benchmark_name] = int(o3_grouped['total_samples'].iloc[0])
        
        # Sort by reasoning effort level
        effort_order = {
            "Low reasoning effort": 0,
            "Medium reasoning effort": 1,
            "High reasoning effort": 2
        }
        
        o3_grouped['order'] = o3_grouped['reasoning_category'].map(effort_order)
        o3_grouped = o3_grouped.sort_values('order')
        
        # First plot the connecting line
        ax.plot(
            o3_grouped['tokens_per_question'],
            o3_grouped['mean_accuracy'] * 100,
            color=o3_color,
            linewidth=2,
            alpha=0.8
        )
        
        # Then plot each point separately with the appropriate marker
        for _, row in o3_grouped.iterrows():
            category = row['reasoning_category']
            marker = o3_markers.get(category, "o")
            
            ax.errorbar(
                row['tokens_per_question'], 
                row['mean_accuracy'] * 100,  # Convert to percentage
                yerr=row['std_accuracy'] * 100,  # Convert to percentage
                fmt=marker, 
                markersize=8, 
                capsize=3, 
                capthick=1.5,
                elinewidth=1.5,
                color=o3_color,
                markeredgecolor=o3_color,  # Set marker edge color to match fill
                label=f'o3-mini ({category})' if _ == 0 else "",
                alpha=0.8
            )
    
    # Set title and axis labels
    ax.set_title(display_name, fontsize=16)
    
    # Always use log scale for x-axis
    ax.set_xscale('log')
    
    # Set x-axis limits - start at 100 as requested
    ax.set_xlim(1, 100000)
    
    # Format x-axis tick labels
    def format_ticks(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    
    # Set explicit x-axis ticks
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
    
    # Set y-axis limits with some padding to ensure error bars aren't cut off
    if not (claude_data.empty and o3_data.empty):
        min_acc = min(
            (claude_data['mean_accuracy'].min() * 100 - 2 * claude_data['std_accuracy'].max() * 100) if not claude_data.empty else float('inf'),
            (o3_data['mean_accuracy'].min() * 100 - 2 * o3_data['std_accuracy'].max() * 100) if not o3_data.empty else float('inf')
        )
        max_acc = max(
            (claude_data['mean_accuracy'].max() * 100 + 2 * claude_data['std_accuracy'].max() * 100) if not claude_data.empty else 0,
            (o3_data['mean_accuracy'].max() * 100 + 2 * o3_data['std_accuracy'].max() * 100) if not o3_data.empty else 0
        )
        
        y_padding = (max_acc - min_acc) * 0.15
        ax.set_ylim(max(0, min_acc - y_padding), min(100, max_acc + y_padding))
    
    # Add grid with custom style - more subtle
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

def create_multi_benchmark_plot(all_data, output_file, csv_output=None):
    """Create a single row plot showing all benchmarks."""
    # Set up the plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    sns.set_style("whitegrid")
    
    # Create figure - make it wider to accommodate single row
    fig = plt.figure(figsize=(20, 5), dpi=300)
    
    # Create GridSpec with configurable spacing
    gs = fig.add_gridspec(1, 4, wspace=0.3)
    
    # Create axes using GridSpec
    axs = []
    for i in range(4):
        axs.append(fig.add_subplot(gs[0, i]))
    
    # Define benchmark order and names
    benchmarks = ['gpqa', 'lab-bench-cloningscenarios', 'lab-bench-protocolqa', 'vct']
    
    # Keep track of sample counts
    sample_counts = {}
    
    # Create a list to store all the data for CSV export
    all_plotted_data = []
    
        # Plot each benchmark in its subplot
    for i, benchmark_name in enumerate(benchmarks):
        if benchmark_name not in all_data:
            print(f"No data found for benchmark: {benchmark_name}")
            continue
            
        claude_data, o3_data = all_data[benchmark_name]
        plot_single_benchmark(axs[i], claude_data, o3_data, benchmark_name, sample_counts)
        
        # Add x-axis label to all subplots
        axs[i].set_xlabel("Mean Output Tokens per Question", fontsize=14)
        
        # Only add y-axis label to leftmost subplot
        if i == 0:
            axs[i].set_ylabel("Accuracy", fontsize=14)
        
        # Process data for CSV export if requested
        if csv_output:
            # Process Claude data
            if not claude_data.empty:
                claude_grouped = claude_data.groupby('reasoning_category').agg({
                    'mean_output_tokens': 'mean',
                    'mean_accuracy': 'mean',
                    'std_accuracy': 'mean',
                    'total_samples': 'first'
                }).reset_index()
                
                claude_grouped['tokens_per_question'] = claude_grouped['mean_output_tokens'] / claude_grouped['total_samples']
                
                for _, row in claude_grouped.iterrows():
                    all_plotted_data.append({
                        'benchmark': benchmark_name,
                        'model': 'Claude 3.7 Sonnet',
                        'reasoning_category': row['reasoning_category'],
                        'mean_accuracy': row['mean_accuracy'] * 100,  # Convert to percentage
                        'std_accuracy': row['std_accuracy'] * 100,    # Convert to percentage
                        'tokens_per_question': row['tokens_per_question'],
                        'mean_output_tokens': row['mean_output_tokens'],
                        'total_samples': row['total_samples']
                    })
            
            # Process o3-mini data
            if not o3_data.empty:
                o3_grouped = o3_data.groupby('reasoning_category').agg({
                    'mean_output_tokens': 'mean',
                    'mean_accuracy': 'mean',
                    'std_accuracy': 'mean',
                    'total_samples': 'first'
                }).reset_index()
                
                o3_grouped['tokens_per_question'] = o3_grouped['mean_output_tokens'] / o3_grouped['total_samples']
                
                for _, row in o3_grouped.iterrows():
                    all_plotted_data.append({
                        'benchmark': benchmark_name,
                        'model': 'o3-mini',
                        'reasoning_category': row['reasoning_category'],
                        'mean_accuracy': row['mean_accuracy'] * 100,  # Convert to percentage
                        'std_accuracy': row['std_accuracy'] * 100,    # Convert to percentage
                        'tokens_per_question': row['tokens_per_question'],
                        'mean_output_tokens': row['mean_output_tokens'],
                        'total_samples': row['total_samples']
                    })
    
    # Save data to CSV if requested
    if csv_output and all_plotted_data:
        df_export = pd.DataFrame(all_plotted_data)
        
        # Map benchmark names to their display names
        benchmark_display = {
            "vct": "VCT",
            "gpqa": "GPQA",
            "lab-bench-cloningscenarios": "CloningScenarios",
            "lab-bench-protocolqa": "ProtocolQA"
        }
        df_export['benchmark_display'] = df_export['benchmark'].map(lambda x: benchmark_display.get(x, x))
        
        # Save to CSV
        df_export.to_csv(csv_output, index=False)
        print(f"Data exported to CSV: {csv_output}")
    
    # Define colors and other styling elements
    claude_color = "#5ab4ac"
    o3_color = "#d8b365"
    markers = ['o', '^', 's']
    claude_labels = ["No reasoning limit specified", "4k reasoning token limit", "16k reasoning token limit"]
    o3_labels = ["Low reasoning effort", "Medium reasoning effort", "High reasoning effort"]
    
    # Create legend elements
    legend_elements = []
    
    # Add Claude elements
    legend_elements.append(plt.Line2D([0], [0], marker=None, color='none', label=f'Claude 3.7 Sonnet', 
                          markerfacecolor='none', markersize=8))
    for i, label in enumerate(claude_labels):
        legend_elements.append(plt.Line2D([0], [0], marker=markers[i], color='none', label=f'    {label}', 
                              markerfacecolor=claude_color, markeredgecolor=claude_color, markersize=8))
    
    # Add o3-mini elements
    legend_elements.append(plt.Line2D([0], [0], marker=None, color='none', label=f'o3-mini', 
                          markerfacecolor='none', markersize=8))
    for i, label in enumerate(o3_labels):
        legend_elements.append(plt.Line2D([0], [0], marker=markers[i], color='none', label=f'    {label}', 
                              markerfacecolor=o3_color, markeredgecolor=o3_color, markersize=8))
    
    # Position the legend - adjust for single row layout
    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=2,
        fontsize=14,
        frameon=True,
        columnspacing=1.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, -0.3)  # Adjusted to account for single row
    )
    
    # Set different colors for model name labels
    legend_texts = legend.get_texts()
    legend_texts[0].set_color(claude_color)
    legend_texts[0].set_fontweight('bold')
    legend_texts[4].set_color(o3_color)
    legend_texts[4].set_fontweight('bold')
    
    # Add main title - adjust y position for single row layout
    fig.suptitle("Effect of Reasoning Effort on Model Performance Across Benchmarks", 
                 fontsize=14, fontweight='bold', y=1.05)
    
    # Adjust layout for legend
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate reasoning effort vs accuracy plots.')
    parser.add_argument('input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='reasoning_plot.png', help='Output image file path')
    parser.add_argument('--csv', type=str, help='Optional CSV output file for the plotted data')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle CSV output path if provided
    csv_output = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_output = csv_path
    
    # Load and prepare data
    df = load_and_prepare_data(args.input)
    
    # Fixed set of benchmarks
    benchmarks = ['gpqa', 'lab-bench-cloningscenarios', 'lab-bench-protocolqa', 'vct']
    
    # Store data for all benchmarks
    all_benchmark_data = {}
    
    # Process each benchmark
    for benchmark in benchmarks:
        claude_data, o3_data = get_reasoning_data(df, benchmark)
        
        if claude_data.empty and o3_data.empty:
            print(f"No data found for benchmark: {benchmark}")
            continue
            
        all_benchmark_data[benchmark] = (claude_data, o3_data)
    
    # Create the combined plot and optionally save CSV
    create_multi_benchmark_plot(all_data=all_benchmark_data, output_file=args.output, csv_output=csv_output)

if __name__ == "__main__":
    main()