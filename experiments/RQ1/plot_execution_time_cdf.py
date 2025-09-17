#!/usr/bin/env python3
"""
Script to generate CDF plot for factory detector execution times
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

def load_execution_times(json_file):
    """Load execution times from evaluation results JSON file"""
    execution_times = []

    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'execution_time_ms' in data:
                    execution_times.append(data['execution_time_ms'])
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

    return np.array(execution_times)

def plot_execution_time_cdf(execution_times, output_path):
    """Plot CDF of execution times with scientific publication style"""

    # Set up scientific publication style
    plt.style.use('default')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times']
    rcParams['font.size'] = 12
    rcParams['axes.linewidth'] = 1.2
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['xtick.minor.width'] = 0.8
    rcParams['ytick.minor.width'] = 0.8
    rcParams['lines.linewidth'] = 2.0
    rcParams['grid.linewidth'] = 0.8
    rcParams['grid.alpha'] = 0.3

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 2))

    # Sort execution times
    sorted_times = np.sort(execution_times)
    n = len(sorted_times)

    # Calculate CDF
    y_values = np.arange(1, n + 1) / n

    # Plot CDF curve
    ax.plot(sorted_times, y_values, 'b-', linewidth=2.5, label='Factory Detector')

    # Calculate percentiles
    p25 = np.percentile(sorted_times, 25)
    p50 = np.percentile(sorted_times, 50)  # median
    p75 = np.percentile(sorted_times, 75)

    # Add vertical lines for percentiles
    ax.axvline(p25, color='red', linestyle='--', linewidth=2, alpha=0.8, label='25th percentile')
    ax.axvline(p50, color='green', linestyle='--', linewidth=2, alpha=0.8, label='50th percentile')
    ax.axvline(p75, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='75th percentile')

    # Add horizontal lines for percentiles
    ax.axhline(0.25, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(0.50, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(0.75, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)

    # Customize axes
    ax.set_xlabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('Cumulative Distribution of Factory Detector Execution Times',
                 fontsize=16, fontweight='bold', pad=20)

    # Set axis limits and scale
    ax.set_xlim(0, max(sorted_times) * 1.05)
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # Add legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Add percentile annotations
    ax.annotate(f'{p25:.1f}ms', xy=(p25, 0.25), xytext=(p25 + max(sorted_times) * 0.1, 0.25),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
                fontsize=10, ha='left', va='center', color='red')

    ax.annotate(f'{p50:.1f}ms', xy=(p50, 0.50), xytext=(p50 + max(sorted_times) * 0.1, 0.50),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.8),
                fontsize=10, ha='left', va='center', color='green')

    ax.annotate(f'{p75:.1f}ms', xy=(p75, 0.75), xytext=(p75 + max(sorted_times) * 0.1, 0.75),
                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.8),
                fontsize=10, ha='left', va='center', color='orange')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')

    # Print statistics
    print(f"Execution Time Statistics:")
    print(f"  Total contracts analyzed: {n:,}")
    print(f"  Mean execution time: {np.mean(sorted_times):.2f} ms")
    print(f"  Median (50th percentile): {p50:.2f} ms")
    print(f"  25th percentile: {p25:.2f} ms")
    print(f"  75th percentile: {p75:.2f} ms")
    print(f"  90th percentile: {np.percentile(sorted_times, 90):.2f} ms")
    print(f"  95th percentile: {np.percentile(sorted_times, 95):.2f} ms")
    print(f"  99th percentile: {np.percentile(sorted_times, 99):.2f} ms")
    print(f"  Maximum execution time: {np.max(sorted_times):.2f} ms")

    return p25, p50, p75

def generate_performance_text(p25, p50, p75):
    """Generate performance analysis text for the paper"""
    text = f"""\\textbf{{Execution performance.}} Our factory detector demonstrates excellent computational efficiency across the entire evaluation dataset. The execution time analysis reveals that 25% of contracts can be analyzed within {p25:.1f} milliseconds, while 50% of contracts complete analysis in {p50:.1f} milliseconds or less. Furthermore, 75% of all contracts achieve complete factory detection within {p75:.1f} milliseconds, demonstrating the scalability of our approach for large-scale blockchain analysis. The cumulative distribution shows a favorable performance profile with rapid analysis times for the majority of contracts, ensuring practical applicability for real-world factory detection tasks across extensive contract datasets."""

    return text

def main():
    # File paths
    json_file = '/Users/mac/ResearchSpace/TOSEM/experiments/RQ1/factory_detector_evaluation_results.json'
    output_path = '/Users/mac/ResearchSpace/TOSEM/experiments/RQ1/execution_time_cdf.pdf'

    print("Loading execution times from evaluation results...")
    execution_times = load_execution_times(json_file)

    if len(execution_times) == 0:
        print("Error: No execution times found in the data file")
        return

    print(f"Loaded {len(execution_times):,} execution time measurements")

    print("Generating CDF plot...")
    p25, p50, p75 = plot_execution_time_cdf(execution_times, output_path)

    print(f"CDF plot saved to: {output_path}")
    print(f"PNG version saved to: {output_path.replace('.pdf', '.png')}")

    print("\nGenerating performance text for paper...")
    performance_text = generate_performance_text(p25, p50, p75)

    # Save performance text to file
    text_file = '/Users/mac/ResearchSpace/TOSEM/experiments/RQ1/execution_performance_text.txt'
    with open(text_file, 'w') as f:
        f.write(performance_text)

    print(f"Performance text saved to: {text_file}")
    print("\nPerformance text:")
    print(performance_text)

if __name__ == "__main__":
    main()
