import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path

def create_tokens_vs_accuracy_plot(output_file):
    # Set up the plot style to match other plots in the preprint
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    sns.set_style("whitegrid")
    
    # Data for Claude 3.7 Sonnet
    claude_tokens_per_question = [9.9, 1207.9, 4514.9]  # Mean tokens per question
    claude_accuracy = [40.9, 37.5, 37.1]  # Mean accuracy (%)
    claude_std = [2.4, 3.1, 2.9]  # Standard deviation
    claude_labels = ["No reasoning", "4k reasoning token limit", "16k reasoning token limit"]

    # Data for o3-mini
    o3_tokens_per_question = [316.8, 1336.6, 4465.3]  # Mean tokens per question
    o3_accuracy = [31.0, 37.1, 40.6]  # Mean accuracy (%)
    o3_std = [2.0, 3.0, 3.9]  # Standard deviation
    o3_labels = ["Low reasoning effort", "Medium reasoning effort", "High reasoning effort"]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Define colors to match other plots
    claude_color = "#5ab4ac"  # Light blue-green from 5-shot color
    o3_color = "#d8b365"      # Light brown from CoT color

    # Plot data for Claude 3.7 Sonnet with error bars
    ax.errorbar(
        claude_tokens_per_question, 
        claude_accuracy, 
        yerr=claude_std,
        fmt='o-', 
        linewidth=2, 
        markersize=8, 
        capsize=3, 
        capthick=1.5,
        elinewidth=1.5,
        color=claude_color, 
        label='Claude 3.7 Sonnet',
        alpha=0.8
    )

    # Plot data for o3-mini with error bars
    ax.errorbar(
        o3_tokens_per_question, 
        o3_accuracy, 
        yerr=o3_std,
        fmt='s-', 
        linewidth=2, 
        markersize=8, 
        capsize=3, 
        capthick=1.5,
        elinewidth=1.5,
        color=o3_color, 
        label='o3-mini',
        alpha=0.8
    )

    # Add annotations for each point
    annotation_offset = {
        "No reasoning": (5, 5),
        "4k reasoning token limit": (5, 5),
        "16k reasoning token limit": (5, 5),
        "Low reasoning effort": (5, -15),
        "Medium reasoning effort": (5, -15),
        "High reasoning effort": (5, -15)
    }

    # Add annotations for Claude data points
    for x, y, label in zip(claude_tokens_per_question, claude_accuracy, claude_labels):
        offset = annotation_offset[label]
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=claude_color,
            alpha=0.9
        )

    # Add annotations for o3-mini data points
    for x, y, label in zip(o3_tokens_per_question, o3_accuracy, o3_labels):
        offset = annotation_offset[label]
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=o3_color,
            alpha=0.9
        )

    # Set title and axis labels
    ax.set_title("VCT No-Images: Accuracy vs Tokens per Question", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean Output Tokens per Question", fontsize=12)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)

    # Set x-axis to log scale for better visualization
    ax.set_xlim(8, 6000)
    
    # Format x-axis tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,d}"))
    
    # Set y-axis limits to focus on the relevant range
    ax.set_ylim(25, 45)
    
    # Add grid with custom style
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend with style matching other plots
    legend = ax.legend(loc='best', frameon=True, fontsize=10)
    legend.get_frame().set_alpha(1)
    
    # Add note about the data
    fig.text(
        0.5, 
        0.01, 
        "VCT no-images has 101 questions. Accuracy shown with standard deviation error bars.",
        ha='center', 
        fontsize=9, 
        fontstyle='italic'
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate tokens vs accuracy plot for VCT no-images.')
    parser.add_argument('--output', type=str, default='vct_reasoning_effort_vs_accuracy.png', 
                        help='Output image file path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_tokens_vs_accuracy_plot(args.output)

if __name__ == "__main__":
    main()