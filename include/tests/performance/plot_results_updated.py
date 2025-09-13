"""
DTW Benchmark Results Plotter - Updated for Separated Benchmarks with CUDA
Handles separate sequential, OpenMP, MPI, and CUDA benchmark files for accurate speedup calculations
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

def load_baseline_data(filename='dtw_baseline_sequential.csv'):
    """Load pure sequential baseline results"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} baseline results from {filename}")

        # Normalize columns
        if 'Constraint' not in df:
            df['Constraint'] = 'None'
        else:
            df['Constraint'] = df['Constraint'].fillna('None')

        # Ensure column consistency
        df['Backend'] = 'Sequential'
        df['Workers'] = 1

        return df
    except FileNotFoundError:
        print(f"Warning: Could not find baseline file {filename}")
        return None

def load_openmp_data(filename='dtw_benchmark_openmp.csv'):
    """Load OpenMP benchmark results"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} OpenMP results from {filename}")

        # Normalize columns
        if 'Constraint' not in df:
            df['Constraint'] = 'None'
        else:
            df['Constraint'] = df['Constraint'].fillna('None')

        # Rename columns for consistency
        df['Workers'] = df['Threads']

        return df
    except FileNotFoundError:
        print(f"Warning: Could not find OpenMP file {filename}")
        return None

def load_mpi_data(pattern='dtw_benchmark_mpi_*.csv'):
    """Load MPI benchmark results from multiple files"""
    all_data = []

    for filename in glob.glob(pattern):
        try:
            df = pd.read_csv(filename)

            # Normalize columns
            if 'Constraint' not in df:
                df['Constraint'] = 'None'
            else:
                df['Constraint'] = df['Constraint'].fillna('None')
            # Rename columns for consistency
            df['Workers'] = df['Processes']
            print(f"Loaded {len(df)} MPI results from {filename}")
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def load_cuda_data(filename='dtw_benchmark_cuda.csv'):
    """Load CUDA benchmark results"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} CUDA results from {filename}")

        # Normalize columns
        if 'Constraint' not in df:
            df['Constraint'] = 'None'
        else:
            df['Constraint'] = df['Constraint'].fillna('None')

        # Keep TileSize as is for CUDA-specific analysis
        # But add Workers for compatibility with other functions
        df['Workers'] = df['TileSize']

        return df
    except FileNotFoundError:
        print(f"Warning: Could not find CUDA file {filename}")
        return None

def calculate_speedup(baseline_df, benchmark_df, constraint='None'):
    """Calculate speedup relative to pure sequential baseline"""

    # Filter baseline for specific constraint
    baseline_constraint = baseline_df[baseline_df['Constraint'] == constraint]

    # Get sequential baseline times (use 'Sequential' strategy)
    baseline_times = {}
    for _, row in baseline_constraint.iterrows():
        if row['Strategy'] == 'Sequential':
            baseline_times[row['Size']] = row['Time_ms']

    # Check if this is CUDA data by checking column existence once
    is_cuda_data = 'TileSize' in benchmark_df.columns

    # Calculate speedup for benchmark data
    speedups = []  # List to collect all speedup entries
    for _, row in benchmark_df.iterrows():
        if row['Size'] in baseline_times and row['Constraint'] == constraint:
            speedup = baseline_times[row['Size']] / row['Time_ms']

            # Create a dictionary for this entry
            speedup_entry = {
                'Backend': row['Backend'],
                'Size': row['Size'],
                'Constraint': row['Constraint'],
                'Time_ms': row['Time_ms'],
                'Speedup': speedup,
                'Workers': row['Workers']
            }

            # Add TileSize for CUDA data or Workers for other data
            if is_cuda_data:
                speedup_entry['TileSize'] = row['TileSize']
            else:
                # Non-CUDA data
                speedup_entry['Efficiency'] = speedup / row['Workers'] if row['Workers'] > 0 else speedup

            # Append the entry to our list
            speedups.append(speedup_entry)

    # Return a DataFrame created from all entries
    return pd.DataFrame(speedups)
def plot_combined_comparison(baseline_df, openmp_df, mpi_df, cuda_df, constraint='None', output_dir='benchmark_plots'):
    """Plot combined comparison of all backends including CUDA"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Filter baseline
    baseline = baseline_df[(baseline_df['Constraint'] == constraint) &
                           (baseline_df['Strategy'] == 'Sequential')]

    # Colors for different worker counts
    colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728'}

    # Get sizes
    sizes = sorted(baseline['Size'].unique())

    # Plot 1: Execution Time Comparison (Top Left)
    ax1 = axes[0, 0]
    baseline = baseline.sort_values('Size')
    ax1.plot(baseline['Size'], baseline['Time_ms'],
             marker='s', color='#808080', label='Sequential',
             linewidth=2, markersize=8)

    if openmp_df is not None:
        omp_constraint = openmp_df[openmp_df['Constraint'] == constraint]
        for workers in sorted(omp_constraint['Workers'].unique()):
            data = omp_constraint[omp_constraint['Workers'] == workers]
            data = data.sort_values('Size')
            ax1.plot(data['Size'], data['Time_ms'],
                     marker='o', color=colors.get(workers, '#000000'),
                     label=f'OpenMP-{workers}T', linewidth=2, markersize=6,
                     linestyle='--', alpha=0.8)

    if mpi_df is not None:
        mpi_constraint = mpi_df[mpi_df['Constraint'] == constraint]
        for workers in sorted(mpi_constraint['Workers'].unique())[:4]:  # Limit to 1,2,4,8
            data = mpi_constraint[mpi_constraint['Workers'] == workers]
            data = data.sort_values('Size')
            ax1.plot(data['Size'], data['Time_ms'],
                     marker='^', color=colors.get(workers, '#000000'),
                     label=f'MPI-{workers}P', linewidth=2, markersize=6,
                     linestyle=':', alpha=0.8)

    # Add CUDA with tile size 512
    if cuda_df is not None:
        cuda_constraint = cuda_df[(cuda_df['Constraint'] == constraint) &
                                  (cuda_df['TileSize'] == 512)]
        if not cuda_constraint.empty:
            cuda_constraint = cuda_constraint.sort_values('Size')
            ax1.plot(cuda_constraint['Size'], cuda_constraint['Time_ms'],
                     marker='d', color='#FF1493', label='CUDA-512',
                     linewidth=2.5, markersize=8, linestyle='-')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Problem Size (n)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title(f'Execution Time - {constraint}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)

    # Plot 2: OpenMP Speedup (Top Right)
    ax2 = axes[0, 1]
    if openmp_df is not None:
        omp_speedup = calculate_speedup(baseline_df, openmp_df, constraint)

        for workers in sorted(omp_speedup['Workers'].unique()):
            data = omp_speedup[omp_speedup['Workers'] == workers]
            data = data.sort_values('Size')
            ax2.plot(data['Size'], data['Speedup'],
                     marker='o', color=colors.get(workers, '#000000'),
                     label=f'{workers} threads', linewidth=2, markersize=8)

        # Add ideal speedup lines
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        for threads in [2, 4, 8]:
            ax2.axhline(y=threads, color='lightgray', linestyle=':', alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Problem Size (n)', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title(f'OpenMP Speedup - {constraint}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(bottom=0, top=10)

    # Plot 3: MPI Speedup (Bottom Left)
    ax3 = axes[1, 0]
    if mpi_df is not None:
        mpi_speedup = calculate_speedup(baseline_df, mpi_df, constraint)

        for workers in sorted(mpi_speedup['Workers'].unique())[:4]:
            data = mpi_speedup[mpi_speedup['Workers'] == workers]
            data = data.sort_values('Size')
            ax3.plot(data['Size'], data['Speedup'],
                     marker='^', color=colors.get(workers, '#000000'),
                     label=f'{workers} processes', linewidth=2, markersize=8)

        # Add ideal speedup lines
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        for procs in [2, 4, 8]:
            ax3.axhline(y=procs, color='lightgray', linestyle=':', alpha=0.3)

    ax3.set_xscale('log', base=2)
    ax3.set_xlabel('Problem Size (n)', fontsize=12)
    ax3.set_ylabel('Speedup', fontsize=12)
    ax3.set_title(f'MPI Speedup - {constraint}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_ylim(bottom=0, top=10)

    # Plot 4: CUDA Speedup (Bottom Right)
    ax4 = axes[1, 1]
    if cuda_df is not None:
        cuda_speedup = calculate_speedup(baseline_df, cuda_df, constraint)

        if not cuda_speedup.empty and 'TileSize' in cuda_speedup.columns:
            # Plot speedup for tile size 512
            cuda_512 = cuda_speedup[cuda_speedup['TileSize'] == 512]
            if not cuda_512.empty:
                cuda_512 = cuda_512.sort_values('Size')
                ax4.plot(cuda_512['Size'], cuda_512['Speedup'],
                         marker='d', color='#FF1493', label='CUDA Tile 512',
                         linewidth=2.5, markersize=10)

        # Add reference lines
        ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='CPU Baseline')
        ax4.axhline(y=10, color='orange', linestyle=':', alpha=0.3, label='10x speedup')
        ax4.axhline(y=50, color='green', linestyle=':', alpha=0.3, label='50x speedup')
        ax4.axhline(y=100, color='blue', linestyle=':', alpha=0.3, label='100x speedup')

    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Problem Size (n)', fontsize=12)
    ax4.set_ylabel('Speedup', fontsize=12)
    ax4.set_title(f'CUDA Speedup - {constraint}', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_ylim(bottom=0)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/combined_{constraint.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {filename}")
    plt.close()

def plot_cuda_analysis(baseline_df, cuda_df, output_dir='benchmark_plots'):
    """Detailed CUDA performance analysis"""

    if cuda_df is None or cuda_df.empty:
        print("No CUDA data available for analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Tile Size Comparison for No Constraints
    ax1 = axes[0, 0]
    cuda_none = cuda_df[cuda_df['Constraint'] == 'None']

    tile_colors = {32: '#1f77b4', 64: '#ff7f0e', 128: '#2ca02c',
                   256: '#d62728', 512: '#9467bd'}

    for tile_size in sorted(cuda_none['TileSize'].unique()):
        data = cuda_none[cuda_none['TileSize'] == tile_size]
        data = data.sort_values('Size')
        ax1.plot(data['Size'], data['Time_ms'],
                 marker='o', color=tile_colors.get(tile_size, '#000000'),
                 label=f'Tile {tile_size}', linewidth=2, markersize=8)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Problem Size (n)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('CUDA Performance vs Tile Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left')

    # Plot 2: CUDA Speedup for Different Tile Sizes
    ax2 = axes[0, 1]
    cuda_speedup = calculate_speedup(baseline_df, cuda_df, 'None')


    for tile_size in sorted(cuda_speedup['TileSize'].unique()):
        data = cuda_speedup[cuda_speedup['TileSize'] == tile_size]
        data = data.sort_values('Size')
        ax2.plot(data['Size'], data['Speedup'],
                 marker='o', color=tile_colors.get(tile_size, '#000000'),
                 label=f'Tile {tile_size}', linewidth=2, markersize=8)

    # Add reference lines
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='CPU Baseline')
    ax2.axhline(y=10, color='orange', linestyle=':', alpha=0.3)
    ax2.axhline(y=50, color='green', linestyle=':', alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Problem Size (n)', fontsize=12)
    ax2.set_ylabel('Speedup over Sequential CPU', fontsize=12)
    ax2.set_title('CUDA Speedup vs Tile Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(bottom=0)

    # Plot 3: CUDA vs Other Backends (using tile 512)
    ax3 = axes[1, 0]

    # Get CUDA times for tile 512
    cuda_512 = cuda_df[(cuda_df['TileSize'] == 512) & (cuda_df['Constraint'] == 'None')]
    if not cuda_512.empty:
        cuda_512 = cuda_512.sort_values('Size')
        ax3.plot(cuda_512['Size'], cuda_512['Time_ms'],
                 marker='d', color='#FF1493', label='CUDA-512',
                 linewidth=2.5, markersize=10)

    # Add sequential baseline for comparison
    baseline = baseline_df[(baseline_df['Constraint'] == 'None') &
                           (baseline_df['Strategy'] == 'Sequential')]
    baseline = baseline.sort_values('Size')
    ax3.plot(baseline['Size'], baseline['Time_ms'],
             marker='s', color='#808080', label='Sequential CPU',
             linewidth=2, markersize=8)

    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Problem Size (n)', fontsize=12)
    ax3.set_ylabel('Execution Time (ms)', fontsize=12)
    ax3.set_title('CUDA vs Sequential CPU', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(loc='upper left')

    # Plot 4: Constraint Performance Comparison (tile 512)
    ax4 = axes[1, 1]

    constraints = ['None', 'Sakoe-Chiba', 'Itakura', 'FastDTW']
    constraint_colors = {'None': '#1f77b4', 'Sakoe-Chiba': '#ff7f0e',
                         'Itakura': '#2ca02c', 'FastDTW': '#d62728'}

    for constraint in constraints:
        # For FastDTW, any tile size is fine as we only save one
        if constraint == 'FastDTW':
            data = cuda_df[cuda_df['Constraint'] == constraint]
        else:
            data = cuda_df[(cuda_df['Constraint'] == constraint) &
                           (cuda_df['TileSize'] == 512)]

        if not data.empty:
            data = data.sort_values('Size')
            ax4.plot(data['Size'], data['Time_ms'],
                     marker='o', color=constraint_colors.get(constraint, '#000000'),
                     label=constraint, linewidth=2, markersize=8)

    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.set_xlabel('Problem Size (n)', fontsize=12)
    ax4.set_ylabel('Execution Time (ms)', fontsize=12)
    ax4.set_title('CUDA Performance Across Constraints (Tile 512)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(loc='upper left')

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/cuda_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved CUDA analysis plot: {filename}")
    plt.close()

def plot_scaling_efficiency(baseline_df, openmp_df, mpi_df, output_dir='benchmark_plots'):
    """Plot scaling efficiency for different problem sizes"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # OpenMP Efficiency
    if openmp_df is not None:
        omp_speedup = calculate_speedup(baseline_df, openmp_df, 'None')

        # Group by size and calculate average efficiency
        sizes = sorted(omp_speedup['Size'].unique())

        for workers in [2, 4, 8]:
            efficiencies = []
            valid_sizes = []

            for size in sizes:
                data = omp_speedup[(omp_speedup['Size'] == size) &
                                   (omp_speedup['Workers'] == workers)]
                if not data.empty:
                    efficiencies.append(data['Efficiency'].mean() * 100)
                    valid_sizes.append(size)

            if efficiencies:
                ax1.plot(valid_sizes, efficiencies,
                         marker='o', label=f'{workers} threads',
                         linewidth=2, markersize=8)

    ax1.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='Perfect scaling')
    ax1.axhline(y=50, color='orange', linestyle=':', alpha=0.5, label='50% efficiency')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Problem Size (n)', fontsize=12)
    ax1.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax1.set_title('OpenMP Parallel Efficiency', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 120)

    # MPI Efficiency
    if mpi_df is not None:
        mpi_speedup = calculate_speedup(baseline_df, mpi_df, 'None')

        sizes = sorted(mpi_speedup['Size'].unique())

        for workers in [2, 4, 8]:
            efficiencies = []
            valid_sizes = []

            for size in sizes:
                data = mpi_speedup[(mpi_speedup['Size'] == size) &
                                   (mpi_speedup['Workers'] == workers)]
                if not data.empty:
                    efficiencies.append(data['Efficiency'].mean() * 100)
                    valid_sizes.append(size)

            if efficiencies:
                ax2.plot(valid_sizes, efficiencies,
                         marker='^', label=f'{workers} processes',
                         linewidth=2, markersize=8)

    ax2.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='Perfect scaling')
    ax2.axhline(y=50, color='orange', linestyle=':', alpha=0.5, label='50% efficiency')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Problem Size (n)', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('MPI Parallel Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 120)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/scaling_efficiency.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency plot: {filename}")
    plt.close()

def generate_report(baseline_df, openmp_df, mpi_df, cuda_df, output_dir='benchmark_plots'):
    """Generate comprehensive benchmark report including CUDA"""

    report_file = f"{output_dir}/benchmark_report.txt"
    os.makedirs(output_dir, exist_ok=True)

    with open(report_file, 'w') as f:
        f.write("DTW BENCHMARK REPORT - FAIR COMPARISON\n")
        f.write("=" * 60 + "\n\n")

        f.write("METHODOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write("- Sequential baseline compiled WITHOUT OpenMP/MPI/CUDA\n")
        f.write("- OpenMP benchmarks compiled with OpenMP only\n")
        f.write("- MPI benchmarks compiled with MPI only\n")
        f.write("- CUDA benchmarks compiled with CUDA only\n")
        f.write("- This ensures fair speedup calculations\n\n")

        # Analyze each constraint type
        constraints = ['None', 'Sakoe-Chiba', 'Itakura', 'FastDTW']

        for constraint in constraints:
            f.write(f"\n{constraint.upper()} CONSTRAINT\n")
            f.write("-" * 40 + "\n")

            # Get baseline times
            if baseline_df is not None:
                baseline = baseline_df[(baseline_df['Constraint'] == constraint) &
                                       (baseline_df['Strategy'] == 'Sequential')]

                sizes = sorted(baseline['Size'].unique())

                for size in sizes:
                    base_time = baseline[baseline['Size'] == size]['Time_ms'].values
                    if len(base_time) == 0:
                        continue

                    base_time = base_time[0]
                    f.write(f"\nSize 2^{int(np.log2(size))} ({size:5d}):\n")
                    f.write(f"  Sequential:  {base_time:8.2f} ms (baseline)\n")

                    # OpenMP results
                    if openmp_df is not None:
                        omp_data = openmp_df[(openmp_df['Constraint'] == constraint) &
                                             (openmp_df['Size'] == size)]
                        for workers in sorted(omp_data['Workers'].unique()):
                            omp_time = omp_data[omp_data['Workers'] == workers]['Time_ms'].values
                            if len(omp_time) > 0:
                                speedup = base_time / omp_time[0]
                                eff = speedup / workers * 100
                                f.write(f"  OpenMP-{workers}T:  {omp_time[0]:8.2f} ms "
                                        f"({speedup:5.2f}x speedup, {eff:5.1f}% eff)\n")

                    # MPI results
                    if mpi_df is not None:
                        mpi_data = mpi_df[(mpi_df['Constraint'] == constraint) &
                                          (mpi_df['Size'] == size)]
                        for workers in sorted(mpi_data['Workers'].unique())[:4]:
                            mpi_time = mpi_data[mpi_data['Workers'] == workers]['Time_ms'].values
                            if len(mpi_time) > 0:
                                speedup = base_time / mpi_time[0]
                                eff = speedup / workers * 100
                                f.write(f"  MPI-{workers}P:    {mpi_time[0]:8.2f} ms "
                                        f"({speedup:5.2f}x speedup, {eff:5.1f}% eff)\n")

                    # CUDA results (tile 512)
                    if cuda_df is not None:
                        if constraint == 'FastDTW':
                            cuda_data = cuda_df[(cuda_df['Constraint'] == constraint) &
                                                (cuda_df['Size'] == size)]
                        else:
                            cuda_data = cuda_df[(cuda_df['Constraint'] == constraint) &
                                                (cuda_df['Size'] == size) &
                                                (cuda_df['TileSize'] == 512)]

                        if not cuda_data.empty:
                            cuda_time = cuda_data['Time_ms'].values[0]
                            speedup = base_time / cuda_time
                            f.write(f"  CUDA-512:    {cuda_time:8.2f} ms "
                                    f"({speedup:5.2f}x speedup)\n")

        # Summary statistics
        f.write("\n\nSUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n")

        # Calculate average speedups for largest problem size
        if baseline_df is not None:
            largest_size = max(baseline_df['Size'].unique())

            f.write(f"\nLargest problem size (n={largest_size}):\n")
            f.write("-" * 40 + "\n")

            base_time = baseline_df[(baseline_df['Size'] == largest_size) &
                                    (baseline_df['Strategy'] == 'Sequential') &
                                    (baseline_df['Constraint'] == 'None')]['Time_ms'].values

            if len(base_time) > 0:
                base_time = base_time[0]

                if openmp_df is not None:
                    f.write("\nOpenMP Speedups:\n")
                    for workers in [1, 2, 4, 8]:
                        omp_times = openmp_df[(openmp_df['Size'] == largest_size) &
                                              (openmp_df['Workers'] == workers) &
                                              (openmp_df['Constraint'] == 'None')]['Time_ms'].values
                        if len(omp_times) > 0:
                            speedup = base_time / omp_times[0]
                            f.write(f"  {workers} threads: {speedup:.2f}x\n")

                if mpi_df is not None:
                    f.write("\nMPI Speedups:\n")
                    for workers in [1, 2, 4, 8]:
                        mpi_times = mpi_df[(mpi_df['Size'] == largest_size) &
                                           (mpi_df['Workers'] == workers) &
                                           (mpi_df['Constraint'] == 'None')]['Time_ms'].values
                        if len(mpi_times) > 0:
                            speedup = base_time / mpi_times[0]
                            f.write(f"  {workers} processes: {speedup:.2f}x\n")

                if cuda_df is not None:
                    f.write("\nCUDA Speedups:\n")
                    cuda_none = cuda_df[(cuda_df['Size'] == largest_size) &
                                        (cuda_df['Constraint'] == 'None')]
                    for tile in sorted(cuda_none['TileSize'].unique()):
                        cuda_times = cuda_none[cuda_none['TileSize'] == tile]['Time_ms'].values
                        if len(cuda_times) > 0:
                            speedup = base_time / cuda_times[0]
                            f.write(f"  Tile {tile}: {speedup:.2f}x\n")



        # Best performers by problem size
        f.write("\n\nBEST PERFORMER BY PROBLEM SIZE\n")
        f.write("=" * 60 + "\n")

        if baseline_df is not None:
            sizes = sorted(baseline_df['Size'].unique())

            for size in sizes:
                f.write(f"\nSize {size}:\n")

                best_time = float('inf')
                best_backend = "Unknown"

                # Check sequential
                seq_time = baseline_df[(baseline_df['Size'] == size) &
                                       (baseline_df['Strategy'] == 'Sequential') &
                                       (baseline_df['Constraint'] == 'None')]['Time_ms'].values
                if len(seq_time) > 0 and seq_time[0] < best_time:
                    best_time = seq_time[0]
                    best_backend = "Sequential"

                # Check OpenMP
                if openmp_df is not None:
                    omp_times = openmp_df[(openmp_df['Size'] == size) &
                                          (openmp_df['Constraint'] == 'None')]['Time_ms'].min()
                    if not pd.isna(omp_times) and omp_times < best_time:
                        best_time = omp_times
                        best_backend = "OpenMP"

                # Check MPI
                if mpi_df is not None:
                    mpi_times = mpi_df[(mpi_df['Size'] == size) &
                                       (mpi_df['Constraint'] == 'None')]['Time_ms'].min()
                    if not pd.isna(mpi_times) and mpi_times < best_time:
                        best_time = mpi_times
                        best_backend = "MPI"

                # Check CUDA
                if cuda_df is not None:
                    cuda_times = cuda_df[(cuda_df['Size'] == size) &
                                         (cuda_df['Constraint'] == 'None')]['Time_ms'].min()
                    if not pd.isna(cuda_times) and cuda_times < best_time:
                        best_time = cuda_times
                        best_backend = "CUDA"

                f.write(f"  Best: {best_backend} ({best_time:.2f} ms)\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated successfully\n")

    print(f"Saved benchmark report: {report_file}")

def main():
    """Main function to generate all plots from separated benchmarks"""

    print("DTW Benchmark Results Plotter - Fair Comparison Version with CUDA")
    print("-" * 60)

    # Create output directory
    output_dir = 'benchmark_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Load data from separated benchmark files
    print("\nLoading benchmark data...")
    baseline_df = load_baseline_data('dtw_baseline_sequential.csv')
    openmp_df = load_openmp_data('dtw_benchmark_openmp.csv')
    mpi_df = load_mpi_data('dtw_benchmark_mpi_*.csv')
    cuda_df = load_cuda_data('dtw_benchmark_cuda.csv')

    if baseline_df is None:
        print("\nERROR: Baseline data not found!")
        print("Please run the sequential benchmark first:")
        print("  ./benchmark_sequential_pure")
        return

    # Generate plots for each constraint type
    constraints = baseline_df['Constraint'].unique()

    print(f"\nGenerating plots for {len(constraints)} constraint types...")
    for constraint in constraints:
        print(f"  Processing: {constraint}")
        plot_combined_comparison(baseline_df, openmp_df, mpi_df, cuda_df, constraint, output_dir)

    # Generate efficiency plots
    print("\nGenerating scaling efficiency plots...")
    plot_scaling_efficiency(baseline_df, openmp_df, mpi_df, output_dir)

    # Generate CUDA analysis if CUDA data is available
    if cuda_df is not None:
        print("\nGenerating CUDA analysis plots...")
        plot_cuda_analysis(baseline_df, cuda_df, output_dir)

    # Generate comprehensive report
    print("\nGenerating benchmark report...")
    generate_report(baseline_df, openmp_df, mpi_df, cuda_df, output_dir)

    print(f"\nAll plots and reports saved to '{output_dir}/' directory")
    print("\nGenerated files:")
    for constraint in constraints:
        clean_name = constraint.lower().replace(' ', '_').replace('-', '_')
        print(f"  - combined_{clean_name}.png")
    print(f"  - scaling_efficiency.png")
    if cuda_df is not None:
        print(f"  - cuda_analysis.png")
    print(f"  - benchmark_report.txt")

    print("\n" + "=" * 60)
    print("IMPORTANT: Make sure all benchmarks were run:")
    print("1. ./benchmark_sequential_pure (compiled WITHOUT any parallel libs)")
    print("2. ./benchmark_openmp_only (compiled WITH OpenMP only)")
    print("3. mpirun -np N ./benchmark_mpi_only (compiled WITH MPI only)")
    print("4. ./benchmark_cuda_only (compiled WITH CUDA only)")
    print("=" * 60)
    print("\nDone!")

if __name__ == "__main__":
    main()
