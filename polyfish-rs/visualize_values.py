#!/usr/bin/env python3
"""
Generate clean visualizations of value targets from safetensors files.
Creates one image per chart.
"""

from safetensors import safe_open
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import stats
import sys
import os

# All generated images go into the journal/ folder.
JOURNAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journal")


def _group_prefix(filename_base):
    """Use the trailing timestamp in the filename as the group prefix so all
    charts generated from the same file sort together (e.g. 1783069033_*)."""
    parts = filename_base.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return filename_base


def _output_path(group_prefix, prefix):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    return os.path.join(JOURNAL_DIR, f"{group_prefix}_{prefix}.png")


def create_distribution_plot(values_np, filename_base):
    """Create distribution plot showing the actual value targets."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create smooth KDE curve
    kde = stats.gaussian_kde(values_np)
    x_range = np.linspace(-1, 1, 500)
    kde_values = kde(x_range)

    # Plot the smooth distribution curve
    ax.plot(x_range, kde_values, linewidth=2.5, color='steelblue', label='Distribution')
    ax.fill_between(x_range, kde_values, alpha=0.3, color='steelblue')

    ax.set_xlabel('Value Target', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Value Targets', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    # Add statistics text box
    abs_values = np.abs(values_np)
    in_range = np.sum((abs_values >= 0.1) & (abs_values <= 0.5))
    pct = 100 * in_range / len(values_np)

    stats_text = f'Mean: {np.mean(values_np):.3f}\n'
    stats_text += f'Std: {np.std(values_np):.3f}\n'
    stats_text += f'Min: {np.min(values_np):.3f}\n'
    stats_text += f'Max: {np.max(values_np):.3f}\n'
    stats_text += f'\n{pct:.1f}% in target range\n[0.1, 0.5]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    output_file = _output_path(_group_prefix(filename_base), "distribution")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✅ Distribution plot saved: {output_file}")
    plt.close()

def create_histogram_plot(values_np, filename_base):
    """Create histogram plot with bars instead of smooth curve."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create histogram (let matplotlib choose optimal bins)
    ax.hist(values_np, alpha=0.7, color='steelblue', edgecolor='none')

    ax.set_xlabel('Value Target', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Histogram of Value Targets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    # Add statistics text box
    abs_values = np.abs(values_np)
    in_range = np.sum((abs_values >= 0.1) & (abs_values <= 0.5))
    pct = 100 * in_range / len(values_np)

    stats_text = f'Mean: {np.mean(values_np):.3f}\n'
    stats_text += f'Std: {np.std(values_np):.3f}\n'
    stats_text += f'Min: {np.min(values_np):.3f}\n'
    stats_text += f'Max: {np.max(values_np):.3f}\n'
    stats_text += f'\n{pct:.1f}% in target range\n[0.1, 0.5]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    output_file = _output_path(_group_prefix(filename_base), "histogram")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✅ Histogram plot saved: {output_file}")
    plt.close()

def create_timeseries_plot(values_np, filename_base):
    """Create time series plot showing values over sequential samples."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Plot the time series
    ax.plot(values_np, alpha=0.7, linewidth=0.8, color='navy', label='Value targets')

    ax.set_xlabel('Sample Index (Sequential)', fontsize=12)
    ax.set_ylabel('Value Target', fontsize=12)
    ax.set_title('Value Targets Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    # Add statistics in corner
    strong_signals = np.sum(np.abs(values_np) >= 0.3)
    weak_signals = np.sum(np.abs(values_np) < 0.1)
    stats_text = f'Total samples: {len(values_np)}\n'
    stats_text += f'Strong (|v|≥0.3): {strong_signals} ({100*strong_signals/len(values_np):.1f}%)\n'
    stats_text += f'Weak (|v|<0.1): {weak_signals} ({100*weak_signals/len(values_np):.1f}%)'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    output_file = _output_path(_group_prefix(filename_base), "timeseries")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✅ Time series plot saved: {output_file}")
    plt.close()

def create_absolute_distribution_plot(values_np, filename_base):
    """Create line chart showing distribution of absolute values."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get absolute values
    abs_values = np.abs(values_np)

    # Create smooth KDE curve for absolute values
    kde = stats.gaussian_kde(abs_values)
    x_range = np.linspace(0, 1, 500)
    kde_values = kde(x_range)

    # Plot as a line
    ax.plot(x_range, kde_values, linewidth=2.5, color='darkred', label='Distribution')
    ax.fill_between(x_range, kde_values, alpha=0.3, color='darkred')

    ax.set_xlabel('|Value Target|', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Absolute Value Targets', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Add statistics in corner
    strong_signals = np.sum(abs_values >= 0.3)
    weak_signals = np.sum(abs_values < 0.1)
    in_range = np.sum((abs_values >= 0.1) & (abs_values <= 0.5))
    stats_text = f'Total samples: {len(values_np)}\n'
    stats_text += f'Strong (|v|≥0.3): {strong_signals} ({100*strong_signals/len(values_np):.1f}%)\n'
    stats_text += f'Target range (0.1-0.5): {in_range} ({100*in_range/len(values_np):.1f}%)\n'
    stats_text += f'Weak (|v|<0.1): {weak_signals} ({100*weak_signals/len(values_np):.1f}%)'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    output_file = _output_path(_group_prefix(filename_base), "absolute_distribution")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✅ Absolute value distribution plot saved: {output_file}")
    plt.close()

def create_bucketed_bar_chart(values_np, filename_base):
    """Create bar chart showing percentage of values in 4 buckets."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get absolute values
    abs_values = np.abs(values_np)
    total = len(abs_values)

    # Calculate counts for each bucket
    bucket1 = np.sum(abs_values < 0.1)  # Weak
    bucket2 = np.sum((abs_values >= 0.1) & (abs_values < 0.3))  # Moderate
    bucket3 = np.sum((abs_values >= 0.3) & (abs_values < 0.5))  # Strong
    bucket4 = np.sum(abs_values >= 0.5)  # Saturation

    # Calculate percentages
    percentages = [
        100 * bucket1 / total,
        100 * bucket2 / total,
        100 * bucket3 / total,
        100 * bucket4 / total
    ]

    # Bar labels
    labels = ['< 0.1\n(Weak)', '0.1 - 0.3\n(Moderate)', '0.3 - 0.5\n(Strong)', '≥ 0.5\n(Saturation)']
    colors = ['#e74c3c', '#2ecc71', '#f39c12', '#e74c3c']  # red, green, yellow, red

    # Create bars
    bars = ax.bar(range(4), percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add percentage labels on top of bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Value Target Range (|v|)', fontsize=12)
    ax.set_ylabel('Percentage of Samples', fontsize=12)
    ax.set_title('Value Target Distribution by Bucket', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(percentages) * 1.15)  # Add 15% headroom for labels
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample count info
    stats_text = f'Total samples: {total}\n'
    stats_text += f'Weak: {bucket1}\n'
    stats_text += f'Moderate: {bucket2}\n'
    stats_text += f'Strong: {bucket3}\n'
    stats_text += f'Saturation: {bucket4}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    output_file = _output_path(_group_prefix(filename_base), "bucketed")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✅ Bucketed bar chart saved: {output_file}")
    plt.close()

def main(filename):
    """Main function to generate both visualizations."""
    if not os.path.exists(filename):
        print(f"ERROR: File not found: {filename}")
        sys.exit(1)

    print(f"Loading data from: {filename}")

    with safe_open(filename, framework="pt") as f:
        if "values" not in f.keys():
            print("ERROR: No 'values' tensor found in file!")
            sys.exit(1)

        values = f.get_tensor("values")
        values_np = values.cpu().numpy().flatten()

    # Calculate key statistics
    abs_values = np.abs(values_np)
    strong_signals = np.sum(abs_values >= 0.3)
    weak_signals = np.sum(abs_values < 0.1)
    in_range = np.sum((abs_values >= 0.1) & (abs_values <= 0.5))

    print(f"Found {len(values_np)} value samples")
    print(f"Range: [{values_np.min():.4f}, {values_np.max():.4f}]")
    print(f"Mean: {values_np.mean():.4f}, Std: {values_np.std():.4f}")
    print(f"\nSignal Strength:")
    print(f"  Weak (|v|<0.1):        {weak_signals:4d} ({100*weak_signals/len(values_np):5.1f}%)")
    print(f"  Target range (0.1-0.5): {in_range:4d} ({100*in_range/len(values_np):5.1f}%)")
    print(f"  Strong (|v|≥0.3):       {strong_signals:4d} ({100*strong_signals/len(values_np):5.1f}%)")
    print()

    # Extract base filename
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # Generate plots
    print("Generating visualizations...")
    create_distribution_plot(values_np, base_name)
    create_histogram_plot(values_np, base_name)
    create_timeseries_plot(values_np, base_name)
    create_absolute_distribution_plot(values_np, base_name)
    create_bucketed_bar_chart(values_np, base_name)

    print("\n✅ All visualizations complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_values.py <safetensors_file>")
        print("Example: python3 visualize_values.py games_1783069033.safetensors")
        sys.exit(1)

    main(sys.argv[1])
