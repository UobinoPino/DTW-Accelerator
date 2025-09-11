# #!/usr/bin/env python3
# """
# DTW Benchmark Results Plotter - Fixed Version
# Generates execution time and speedup plots from benchmark CSV data
# """
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import sys
# import os
# import warnings
#
# # Suppress matplotlib backend warnings
# warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg.*')
#
# # Set style for better-looking plots
# plt.style.use('seaborn-v0_8-darkgrid')
#
# def load_benchmark_data(filename='dtw_benchmark_results.csv'):
#     """Load benchmark results from CSV file"""
#     try:
#         df = pd.read_csv(filename)
#         print(f"Loaded {len(df)} benchmark results from {filename}")
#
#         # Convert all column names to strings and handle any issues
#         df['Constraint'] = df['Constraint'].astype(str)
#         df['Strategy'] = df['Strategy'].astype(str)
#
#         # Show what constraints we have
#         print(f"Found constraints: {df['Constraint'].unique()}")
#
#         return df
#     except FileNotFoundError:
#         print(f"Error: Could not find {filename}")
#         print("Make sure to run the C++ benchmark first!")
#         sys.exit(1)
#
# def plot_execution_time(df, constraint_type, output_dir='benchmark_plots'):
#     """Plot execution time vs problem size for a specific constraint type"""
#
#     # Filter data for the specific constraint
#     df_filtered = df[df['Constraint'] == constraint_type]
#
#     if df_filtered.empty:
#         print(f"No data found for constraint: {constraint_type}")
#         return
#
#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Get unique strategies
#     strategies = sorted(df_filtered['Strategy'].unique())
#
#     # Define colors and markers for different strategies
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
#     markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
#
#     # Plot each strategy
#     for i, strategy in enumerate(strategies):
#         strategy_data = df_filtered[df_filtered['Strategy'] == strategy]
#         strategy_data = strategy_data.sort_values('Size')
#
#         ax.plot(strategy_data['Size'], strategy_data['Time_ms'],
#                 marker=markers[i % len(markers)],
#                 color=colors[i % len(colors)],
#                 label=strategy,
#                 linewidth=2,
#                 markersize=8)
#
#     # Set logarithmic scales
#     ax.set_xscale('log', base=2)
#     ax.set_yscale('log')
#
#     # Customize plot
#     ax.set_xlabel('Problem Size (n)', fontsize=12)
#     ax.set_ylabel('Execution Time (ms)', fontsize=12)
#
#     # Better title handling
#     if constraint_type == 'None':
#         title = 'No Constraints'
#     else:
#         title = constraint_type
#     ax.set_title(f'DTW Execution Time - {title}', fontsize=14, fontweight='bold')
#
#     # Set x-axis ticks to show powers of 2
#     sizes = sorted(df_filtered['Size'].unique())
#     ax.set_xticks(sizes)
#     ax.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])
#
#     # Add grid
#     ax.grid(True, alpha=0.3, which='both')
#
#     # Add legend
#     ax.legend(loc='upper left', framealpha=0.9)
#
#     # Save plot
#     os.makedirs(output_dir, exist_ok=True)
#     filename = f"{output_dir}/execution_time_{constraint_type.lower().replace(' ', '_').replace('-', '_')}.png"
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Saved execution time plot: {filename}")
#     plt.close()
#
# def plot_speedup(df, constraint_type, output_dir='benchmark_plots'):
#     """Plot speedup vs problem size for a specific constraint type"""
#
#     # Filter data for this constraint
#     df_filtered = df[df['Constraint'] == constraint_type]
#
#     if df_filtered.empty:
#         print(f"No data found for constraint: {constraint_type}")
#         return
#
#     # Get baseline times (Sequential with None constraint)
#     baseline_data = df[(df['Strategy'] == 'Sequential') & (df['Constraint'] == 'None')]
#
#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Get unique strategies
#     strategies = sorted(df_filtered['Strategy'].unique())
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
#     markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
#
#     # Plot each strategy
#     for i, strategy in enumerate(strategies):
#         strategy_data = df_filtered[df_filtered['Strategy'] == strategy]
#         speedups = []
#         sizes = []
#
#         for _, row in strategy_data.iterrows():
#             size = row['Size']
#             baseline_time = baseline_data[baseline_data['Size'] == size]['Time_ms'].values
#
#             if len(baseline_time) > 0:
#                 speedup = baseline_time[0] / row['Time_ms']
#                 speedups.append(speedup)
#                 sizes.append(size)
#
#         if speedups:
#             # Sort by size
#             sorted_indices = np.argsort(sizes)
#             sizes = [sizes[j] for j in sorted_indices]
#             speedups = [speedups[j] for j in sorted_indices]
#
#             ax.plot(sizes, speedups,
#                     marker=markers[i % len(markers)],
#                     color=colors[i % len(colors)],
#                     label=strategy,
#                     linewidth=2,
#                     markersize=8)
#
#     # Add reference lines
#     ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
#
#     # Add theoretical max speedup line
#     if constraint_type == 'None':
#         import multiprocessing
#         num_cores = multiprocessing.cpu_count()
#         ax.axhline(y=num_cores, color='red', linestyle=':', alpha=0.5,
#                    label=f'Theoretical Max ({num_cores}x)')
#
#     # Set logarithmic x-scale
#     ax.set_xscale('log', base=2)
#
#     # Customize plot
#     ax.set_xlabel('Problem Size (n)', fontsize=12)
#     ax.set_ylabel('Speedup vs Sequential (No Constraints)', fontsize=12)
#
#     # Better title handling
#     if constraint_type == 'None':
#         title = 'No Constraints'
#     else:
#         title = constraint_type
#     ax.set_title(f'DTW Speedup - {title}', fontsize=14, fontweight='bold')
#
#     # Set x-axis ticks
#     sizes_unique = sorted(set(sizes)) if sizes else [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
#     ax.set_xticks(sizes_unique)
#     ax.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes_unique])
#
#     # Add grid
#     ax.grid(True, alpha=0.3)
#
#     # Add legend
#     ax.legend(loc='best', framealpha=0.9)
#
#     # Save plot
#     os.makedirs(output_dir, exist_ok=True)
#     filename = f"{output_dir}/speedup_{constraint_type.lower().replace(' ', '_').replace('-', '_')}.png"
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Saved speedup plot: {filename}")
#     plt.close()
#
# def plot_comparison_summary(df, output_dir='benchmark_plots'):
#     """Create a summary comparison plot across all constraints"""
#
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle('DTW Performance Comparison Summary', fontsize=16, fontweight='bold')
#
#     constraints = ['None', 'Sakoe-Chiba', 'Itakura', 'FastDTW']
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
#
#     for idx, constraint in enumerate(constraints):
#         ax = axes[idx // 2, idx % 2]
#
#         df_filtered = df[df['Constraint'] == constraint]
#         if df_filtered.empty:
#             ax.text(0.5, 0.5, f'No data for {constraint}',
#                     ha='center', va='center', transform=ax.transAxes)
#             continue
#
#         strategies = sorted(df_filtered['Strategy'].unique())
#
#         for i, strategy in enumerate(strategies):
#             strategy_data = df_filtered[df_filtered['Strategy'] == strategy]
#             strategy_data = strategy_data.sort_values('Size')
#
#             ax.plot(strategy_data['Size'], strategy_data['Time_ms'],
#                     marker='o', color=colors[i % len(colors)], label=strategy,
#                     linewidth=1.5, markersize=6)
#
#         ax.set_xscale('log', base=2)
#         ax.set_yscale('log')
#         ax.set_xlabel('Problem Size', fontsize=10)
#         ax.set_ylabel('Time (ms)', fontsize=10)
#         ax.set_title(f'{constraint if constraint != "None" else "No Constraints"}', fontsize=12)
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=8, loc='upper left')
#
#     plt.tight_layout()
#
#     # Save plot
#     os.makedirs(output_dir, exist_ok=True)
#     filename = f"{output_dir}/comparison_summary.png"
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Saved summary plot: {filename}")
#     plt.close()
#
# def generate_report(df, output_dir='benchmark_plots'):
#     """Generate a text report with key findings"""
#
#     report_file = f"{output_dir}/benchmark_report.txt"
#     os.makedirs(output_dir, exist_ok=True)
#
#     with open(report_file, 'w') as f:
#         f.write("DTW BENCHMARK REPORT\n")
#         f.write("=" * 50 + "\n\n")
#
#         # Find best strategy for each constraint and size
#         constraints = df['Constraint'].unique()
#         sizes = sorted(df['Size'].unique())
#
#         for constraint in constraints:
#             constraint_str = str(constraint)
#
#             # Format header
#             if constraint_str == 'None':
#                 header = 'NO CONSTRAINTS'
#             elif constraint_str == 'Sakoe-Chiba':
#                 header = 'SAKOE-CHIBA CONSTRAINT'
#             elif constraint_str == 'Itakura':
#                 header = 'ITAKURA CONSTRAINT'
#             elif constraint_str == 'FastDTW':
#                 header = 'FASTDTW'
#             else:
#                 header = constraint_str.upper()
#
#             f.write(f"\n{header}\n")
#             f.write("-" * 30 + "\n")
#
#             df_constraint = df[df['Constraint'] == constraint]
#
#             for size in sizes:
#                 df_size = df_constraint[df_constraint['Size'] == size]
#                 if not df_size.empty:
#                     best = df_size.loc[df_size['Time_ms'].idxmin()]
#                     speedup_val = best['Speedup'] if pd.notna(best['Speedup']) else 1.0
#                     f.write(f"Size 2^{int(np.log2(size))} ({size:5d}): {best['Strategy']:10s} "
#                             f"({best['Time_ms']:8.2f} ms, {speedup_val:6.2f}x speedup)\n")
#
#         # Calculate average speedups
#         f.write("\n\nAVERAGE SPEEDUPS (vs Sequential No Constraints)\n")
#         f.write("-" * 50 + "\n")
#
#         # Get baseline times
#         baseline_times = {}
#         baseline_data = df[(df['Strategy'] == 'Sequential') & (df['Constraint'] == 'None')]
#         for _, row in baseline_data.iterrows():
#             baseline_times[row['Size']] = row['Time_ms']
#
#         # Calculate speedups for each strategy
#         strategies = sorted(df['Strategy'].unique())
#         for strategy in strategies:
#             strategy_data = df[df['Strategy'] == strategy]
#             speedups = []
#
#             for _, row in strategy_data.iterrows():
#                 if row['Size'] in baseline_times:
#                     speedup = baseline_times[row['Size']] / row['Time_ms']
#                     speedups.append(speedup)
#
#             if speedups:
#                 avg_speedup = np.mean(speedups)
#                 f.write(f"{strategy:15s}: {avg_speedup:6.2f}x\n")
#
#         # Add best strategies summary
#         f.write("\n\nBEST STRATEGIES BY PROBLEM SIZE\n")
#         f.write("-" * 50 + "\n")
#
#         for size in sizes:
#             f.write(f"\nSize 2^{int(np.log2(size))} ({size}):\n")
#
#             # Find best for each constraint at this size
#             for constraint in ['None', 'Sakoe-Chiba', 'Itakura', 'FastDTW']:
#                 df_filtered = df[(df['Constraint'] == constraint) & (df['Size'] == size)]
#                 if not df_filtered.empty:
#                     best = df_filtered.loc[df_filtered['Time_ms'].idxmin()]
#                     f.write(f"  {constraint:12s}: {best['Strategy']:10s} ({best['Time_ms']:.2f} ms)\n")
#
#         f.write("\n" + "=" * 50 + "\n")
#         f.write("Report generated successfully\n")
#
#     print(f"Saved benchmark report: {report_file}")
#
# def main():
#     """Main function to generate all plots"""
#
#     print("DTW Benchmark Results Plotter")
#     print("-" * 40)
#
#     # Load data
#     df = load_benchmark_data()
#
#     # Create output directory
#     output_dir = 'benchmark_plots'
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Generate plots for each constraint type
#     constraints = df['Constraint'].unique()
#
#     print(f"\nGenerating plots for {len(constraints)} constraint types...")
#
#     for constraint in constraints:
#         print(f"\nProcessing: {constraint}")
#         plot_execution_time(df, constraint, output_dir)
#         plot_speedup(df, constraint, output_dir)
#
#     # Generate summary plots
#     print("\nGenerating summary plots...")
#     plot_comparison_summary(df, output_dir)
#
#     # Generate report
#     print("\nGenerating benchmark report...")
#     generate_report(df, output_dir)
#
#     print(f"\nAll plots and reports saved to '{output_dir}/' directory")
#     print("\nPlot files generated:")
#     for constraint in constraints:
#         clean_name = constraint.lower().replace(' ', '_').replace('-', '_')
#         print(f"  - execution_time_{clean_name}.png")
#         print(f"  - speedup_{clean_name}.png")
#     print(f"  - comparison_summary.png")
#     print(f"  - benchmark_report.txt")
#     print("\nDone!")
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
DTW Benchmark Results Plotter - Fixed and Extended Version
Generates execution time, speedup, and scaling plots from benchmark CSV data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings
import glob

# Suppress matplotlib backend warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg.*')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def load_benchmark_data(filename='dtw_benchmark_results.csv'):
    """Load benchmark results from CSV file"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} benchmark results from {filename}")

        # Convert all column names to strings and handle any issues
        df['Constraint'] = df['Constraint'].astype(str)
        df['Strategy'] = df['Strategy'].astype(str)

        # Show what constraints we have
        print(f"Found constraints: {df['Constraint'].unique()}")

        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        print("Make sure to run the C++ benchmark first!")
        return None

def load_scaling_data(pattern='dtw_scaling_results*.csv'):
    """Load scaling benchmark results from CSV files"""
    all_data = []

    for filename in glob.glob(pattern):
        try:
            df = pd.read_csv(filename)
            print(f"Loaded {len(df)} scaling results from {filename}")
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def plot_execution_time(df, constraint_type, output_dir='benchmark_plots'):
    """Plot execution time vs problem size for a specific constraint type"""

    # Filter data for the specific constraint
    df_filtered = df[df['Constraint'] == constraint_type]

    if df_filtered.empty:
        print(f"No data found for constraint: {constraint_type}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique strategies
    strategies = sorted(df_filtered['Strategy'].unique())

    # Define colors and markers for different strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    # Plot each strategy
    for i, strategy in enumerate(strategies):
        strategy_data = df_filtered[df_filtered['Strategy'] == strategy]
        strategy_data = strategy_data.sort_values('Size')

        ax.plot(strategy_data['Size'], strategy_data['Time_ms'],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=strategy,
                linewidth=2,
                markersize=8)

    # Set logarithmic scales
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Customize plot
    ax.set_xlabel('Problem Size (n)', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)

    # Better title handling
    if constraint_type == 'None':
        title = 'No Constraints'
    else:
        title = constraint_type
    ax.set_title(f'DTW Execution Time - {title}', fontsize=14, fontweight='bold')

    # Set x-axis ticks to show powers of 2
    sizes = sorted(df_filtered['Size'].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])

    # Add grid
    ax.grid(True, alpha=0.3, which='both')

    # Add legend
    ax.legend(loc='upper left', framealpha=0.9)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/execution_time_{constraint_type.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved execution time plot: {filename}")
    plt.close()

def plot_speedup(df, constraint_type, output_dir='benchmark_plots'):
    """Plot speedup vs problem size for a specific constraint type"""

    # Filter data for this constraint
    df_filtered = df[df['Constraint'] == constraint_type]

    if df_filtered.empty:
        print(f"No data found for constraint: {constraint_type}")
        return

    # Get baseline times - looking for Sequential strategy with None constraint
    # Try different variations of how "None" might be stored
    baseline_data = None
    for none_value in ['None', 'NONE', 'none', 'nan', None]:
        baseline_data = df[(df['Strategy'] == 'Sequential') & (df['Constraint'] == none_value)]
        if not baseline_data.empty:
            break

    if baseline_data is None or baseline_data.empty:
        print(f"Warning: No baseline data found for speedup calculation")
        # Try to use Sequential from same constraint as baseline
        baseline_data = df_filtered[df_filtered['Strategy'] == 'Sequential']
        if baseline_data.empty:
            print(f"Cannot calculate speedup without baseline data")
            return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique strategies
    strategies = sorted(df_filtered['Strategy'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    # Plot each strategy
    for i, strategy in enumerate(strategies):
        strategy_data = df_filtered[df_filtered['Strategy'] == strategy]
        speedups = []
        sizes = []

        for _, row in strategy_data.iterrows():
            size = row['Size']
            baseline_time = baseline_data[baseline_data['Size'] == size]['Time_ms'].values

            if len(baseline_time) > 0:
                speedup = baseline_time[0] / row['Time_ms']
                speedups.append(speedup)
                sizes.append(size)

        if speedups:
            # Sort by size
            sorted_indices = np.argsort(sizes)
            sizes = [sizes[j] for j in sorted_indices]
            speedups = [speedups[j] for j in sorted_indices]

            ax.plot(sizes, speedups,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=strategy,
                    linewidth=2,
                    markersize=8)

    # Add reference lines
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')

    # Add theoretical max speedup line
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    ax.axhline(y=num_cores, color='red', linestyle=':', alpha=0.5,
               label=f'Theoretical Max ({num_cores}x)')

    # Set logarithmic x-scale
    ax.set_xscale('log', base=2)

    # Customize plot
    ax.set_xlabel('Problem Size (n)', fontsize=12)
    ax.set_ylabel('Speedup vs Sequential (No Constraints)', fontsize=12)

    # Better title handling
    if constraint_type == 'None':
        title = 'No Constraints'
    else:
        title = constraint_type
    ax.set_title(f'DTW Speedup - {title}', fontsize=14, fontweight='bold')

    # Set x-axis ticks
    sizes_unique = sorted(set(sizes)) if sizes else [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    ax.set_xticks(sizes_unique)
    ax.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes_unique])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='best', framealpha=0.9)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/speedup_{constraint_type.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved speedup plot: {filename}")
    plt.close()

def plot_openmp_scaling(df, output_dir='benchmark_plots'):
    """Plot OpenMP scaling performance"""

    if df is None or df.empty:
        print("No scaling data available for OpenMP")
        return

    # Filter for OpenMP and Sequential data
    df_omp = df[df['Backend'].isin(['OpenMP', 'Sequential'])]

    if df_omp.empty:
        print("No OpenMP scaling data found")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for different thread counts
    colors = {'Sequential': '#808080', 1: '#1f77b4', 2: '#ff7f0e',
              4: '#2ca02c', 8: '#d62728'}
    markers = {'Sequential': 's', 1: 'o', 2: '^', 4: 'D', 8: 'v'}

    # Get unique problem sizes
    sizes = sorted(df_omp['Size'].unique())

    # Plot 1: Execution Time
    for workers in sorted(df_omp['Workers'].unique()):
        if workers == 1 and 'Sequential' in df_omp['Backend'].values:
            # Use Sequential for 1 worker
            data = df_omp[(df_omp['Backend'] == 'Sequential')]
            label = 'Sequential'
        else:
            data = df_omp[(df_omp['Backend'] == 'OpenMP') & (df_omp['Workers'] == workers)]
            label = f'{workers} threads'

        if not data.empty:
            data = data.sort_values('Size')
            ax1.plot(data['Size'], data['Time_ms'],
                     marker=markers.get(workers, 'o'),
                     color=colors.get(workers, '#000000'),
                     label=label,
                     linewidth=2,
                     markersize=8)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Problem Size (n)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('OpenMP Scaling - Execution Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', framealpha=0.9)

    # Plot 2: Speedup
    baseline = df_omp[df_omp['Backend'] == 'Sequential']

    for workers in [1, 2, 4, 8]:
        data = df_omp[(df_omp['Backend'] == 'OpenMP') & (df_omp['Workers'] == workers)]
        if data.empty:
            continue

        speedups = []
        sizes_with_speedup = []

        for size in sizes:
            base_time = baseline[baseline['Size'] == size]['Time_ms'].values
            omp_time = data[data['Size'] == size]['Time_ms'].values

            if len(base_time) > 0 and len(omp_time) > 0:
                speedup = base_time[0] / omp_time[0]
                speedups.append(speedup)
                sizes_with_speedup.append(size)

        if speedups:
            ax2.plot(sizes_with_speedup, speedups,
                     marker=markers.get(workers, 'o'),
                     color=colors.get(workers, '#000000'),
                     label=f'{workers} threads',
                     linewidth=2,
                     markersize=8)

    # Add ideal speedup lines
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for threads in [2, 4, 8]:
        ax2.axhline(y=threads, color='lightgray', linestyle=':', alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Problem Size (n)', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('OpenMP Scaling - Speedup', fontsize=14, fontweight='bold')
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/openmp_scaling.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved OpenMP scaling plot: {filename}")
    plt.close()

def plot_mpi_scaling(df, output_dir='benchmark_plots'):
    """Plot MPI scaling performance"""

    if df is None or df.empty:
        print("No scaling data available for MPI")
        return

    # Filter for MPI and Sequential data
    df_mpi = df[df['Backend'].isin(['MPI', 'Sequential'])]

    if df_mpi.empty:
        print("No MPI scaling data found")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for different process counts
    colors = {'Sequential': '#808080', 1: '#1f77b4', 2: '#ff7f0e',
              4: '#2ca02c', 8: '#d62728'}
    markers = {'Sequential': 's', 1: 'o', 2: '^', 4: 'D', 8: 'v'}

    # Get unique problem sizes
    sizes = sorted(df_mpi['Size'].unique())

    # Plot 1: Execution Time
    # First plot sequential
    seq_data = df_mpi[df_mpi['Backend'] == 'Sequential']
    if not seq_data.empty:
        seq_data = seq_data.sort_values('Size')
        ax1.plot(seq_data['Size'], seq_data['Time_ms'],
                 marker=markers['Sequential'],
                 color=colors['Sequential'],
                 label='Sequential',
                 linewidth=2,
                 markersize=8)

    # Then plot MPI results
    for workers in [1, 2, 4, 8]:
        data = df_mpi[(df_mpi['Backend'] == 'MPI') & (df_mpi['Workers'] == workers)]

        if not data.empty:
            data = data.sort_values('Size')
            ax1.plot(data['Size'], data['Time_ms'],
                     marker=markers.get(workers, 'o'),
                     color=colors.get(workers, '#000000'),
                     label=f'{workers} processes',
                     linewidth=2,
                     markersize=8)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Problem Size (n)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('MPI Scaling - Execution Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', framealpha=0.9)

    # Plot 2: Speedup
    baseline = df_mpi[df_mpi['Backend'] == 'Sequential']

    for workers in [1, 2, 4, 8]:
        data = df_mpi[(df_mpi['Backend'] == 'MPI') & (df_mpi['Workers'] == workers)]
        if data.empty:
            continue

        speedups = []
        sizes_with_speedup = []

        for size in sizes:
            base_time = baseline[baseline['Size'] == size]['Time_ms'].values
            mpi_time = data[data['Size'] == size]['Time_ms'].values

            if len(base_time) > 0 and len(mpi_time) > 0:
                speedup = base_time[0] / mpi_time[0]
                speedups.append(speedup)
                sizes_with_speedup.append(size)

        if speedups:
            ax2.plot(sizes_with_speedup, speedups,
                     marker=markers.get(workers, 'o'),
                     color=colors.get(workers, '#000000'),
                     label=f'{workers} processes',
                     linewidth=2,
                     markersize=8)

    # Add ideal speedup lines
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for procs in [2, 4, 8]:
        ax2.axhline(y=procs, color='lightgray', linestyle=':', alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Problem Size (n)', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('MPI Scaling - Speedup', fontsize=14, fontweight='bold')
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f'2^{int(np.log2(s))}' for s in sizes])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/mpi_scaling.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved MPI scaling plot: {filename}")
    plt.close()

def plot_comparison_summary(df, output_dir='benchmark_plots'):
    """Create a summary comparison plot across all constraints"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DTW Performance Comparison Summary', fontsize=16, fontweight='bold')

    # Try different variations of "None"
    constraints = []
    for c in df['Constraint'].unique():
        if c not in constraints:
            constraints.append(c)

    # Ensure we have exactly 4 constraints for the 2x2 grid
    while len(constraints) < 4:
        constraints.append(None)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for idx, constraint in enumerate(constraints[:4]):
        ax = axes[idx // 2, idx % 2]

        if constraint is None:
            ax.text(0.5, 0.5, 'No data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Empty')
            continue

        df_filtered = df[df['Constraint'] == constraint]
        if df_filtered.empty:
            ax.text(0.5, 0.5, f'No data for {constraint}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        strategies = sorted(df_filtered['Strategy'].unique())

        for i, strategy in enumerate(strategies):
            strategy_data = df_filtered[df_filtered['Strategy'] == strategy]
            strategy_data = strategy_data.sort_values('Size')

            ax.plot(strategy_data['Size'], strategy_data['Time_ms'],
                    marker='o', color=colors[i % len(colors)], label=strategy,
                    linewidth=1.5, markersize=6)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Problem Size', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(f'{constraint if constraint != "None" else "No Constraints"}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/comparison_summary.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {filename}")
    plt.close()

def generate_report(df, df_scaling, output_dir='benchmark_plots'):
    """Generate a text report with key findings"""

    report_file = f"{output_dir}/benchmark_report.txt"
    os.makedirs(output_dir, exist_ok=True)

    with open(report_file, 'w') as f:
        f.write("DTW BENCHMARK REPORT\n")
        f.write("=" * 50 + "\n\n")

        if df is not None:
            # Original benchmark report
            constraints = df['Constraint'].unique()
            sizes = sorted(df['Size'].unique())

            for constraint in constraints:
                constraint_str = str(constraint)

                if constraint_str == 'None':
                    header = 'NO CONSTRAINTS'
                elif constraint_str == 'Sakoe-Chiba':
                    header = 'SAKOE-CHIBA CONSTRAINT'
                elif constraint_str == 'Itakura':
                    header = 'ITAKURA CONSTRAINT'
                elif constraint_str == 'FastDTW':
                    header = 'FASTDTW'
                else:
                    header = constraint_str.upper()

                f.write(f"\n{header}\n")
                f.write("-" * 30 + "\n")

                df_constraint = df[df['Constraint'] == constraint]

                for size in sizes:
                    df_size = df_constraint[df_constraint['Size'] == size]
                    if not df_size.empty:
                        best = df_size.loc[df_size['Time_ms'].idxmin()]
                        speedup_val = best['Speedup'] if pd.notna(best['Speedup']) else 1.0
                        f.write(f"Size 2^{int(np.log2(size))} ({size:5d}): {best['Strategy']:10s} "
                                f"({best['Time_ms']:8.2f} ms, {speedup_val:6.2f}x speedup)\n")

        if df_scaling is not None:
            # Scaling report
            f.write("\n\nSCALING PERFORMANCE\n")
            f.write("=" * 50 + "\n")

            # OpenMP scaling efficiency
            f.write("\nOpenMP Scaling Efficiency (average across all sizes):\n")
            f.write("-" * 40 + "\n")

            for threads in [1, 2, 4, 8]:
                omp_data = df_scaling[(df_scaling['Backend'] == 'OpenMP') &
                                      (df_scaling['Workers'] == threads)]
                if not omp_data.empty:
                    avg_efficiency = omp_data['Efficiency'].mean() * 100
                    avg_speedup = omp_data['Speedup'].mean()
                    f.write(f"{threads} threads: {avg_speedup:.2f}x speedup, "
                            f"{avg_efficiency:.1f}% efficiency\n")

            # MPI scaling efficiency
            f.write("\nMPI Scaling Efficiency (average across all sizes):\n")
            f.write("-" * 40 + "\n")

            for procs in [1, 2, 4, 8]:
                mpi_data = df_scaling[(df_scaling['Backend'] == 'MPI') &
                                      (df_scaling['Workers'] == procs)]
                if not mpi_data.empty:
                    avg_efficiency = mpi_data['Efficiency'].mean() * 100
                    avg_speedup = mpi_data['Speedup'].mean()
                    f.write(f"{procs} processes: {avg_speedup:.2f}x speedup, "
                            f"{avg_efficiency:.1f}% efficiency\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("Report generated successfully\n")

    print(f"Saved benchmark report: {report_file}")

def main():
    """Main function to generate all plots"""

    print("DTW Benchmark Results Plotter")
    print("-" * 40)

    # Create output directory
    output_dir = 'benchmark_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Load comprehensive benchmark data
    df = load_benchmark_data()

    if df is not None:
        # Generate plots for each constraint type
        constraints = df['Constraint'].unique()

        print(f"\nGenerating plots for {len(constraints)} constraint types...")

        for constraint in constraints:
            print(f"\nProcessing: {constraint}")
            plot_execution_time(df, constraint, output_dir)
            plot_speedup(df, constraint, output_dir)

        # Generate summary plots
        print("\nGenerating summary plots...")
        plot_comparison_summary(df, output_dir)

    # Load and plot scaling data
    print("\nLoading scaling benchmark data...")
    df_scaling = load_scaling_data()

    if df_scaling is not None:
        print("\nGenerating scaling plots...")
        plot_openmp_scaling(df_scaling, output_dir)
        plot_mpi_scaling(df_scaling, output_dir)
    else:
        print("No scaling data found. Run scaling benchmarks first.")

    # Generate report
    print("\nGenerating benchmark report...")
    generate_report(df, df_scaling, output_dir)

    print(f"\nAll plots and reports saved to '{output_dir}/' directory")
    print("\nPlot files generated:")

    if df is not None:
        for constraint in df['Constraint'].unique():
            clean_name = constraint.lower().replace(' ', '_').replace('-', '_')
            print(f"  - execution_time_{clean_name}.png")
            print(f"  - speedup_{clean_name}.png")
        print(f"  - comparison_summary.png")

    if df_scaling is not None:
        print(f"  - openmp_scaling.png")
        print(f"  - mpi_scaling.png")

    print(f"  - benchmark_report.txt")
    print("\nDone!")

if __name__ == "__main__":
    main()