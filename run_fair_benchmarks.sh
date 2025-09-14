#!/bin/bash
# Fair DTW Benchmarking Script
# Runs separated benchmarks to avoid cross-contamination between parallel libraries


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_SEQUENTIAL=true
RUN_OPENMP=true
RUN_MPI=true
MPI_PROCS="1 2 4 8"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --no-sequential)
            RUN_SEQUENTIAL=false
            shift
            ;;
        --no-openmp)
            RUN_OPENMP=false
            shift
            ;;
        --no-mpi)
            RUN_MPI=false
            shift
            ;;
        --mpi-procs=*)
            MPI_PROCS="${arg#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-sequential    Skip pure sequential baseline"
            echo "  --no-openmp       Skip OpenMP benchmarks"
            echo "  --no-mpi          Skip MPI benchmarks"
            echo "  --mpi-procs=LIST  MPI process counts (default: '1 2 4 8')"
            echo "  --help            Display this help message"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    FAIR DTW BENCHMARKING SUITE       ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}This script runs separated benchmarks to ensure fair comparisons${NC}"
echo -e "${YELLOW}Each benchmark is compiled without other parallel libraries${NC}"
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Build directory not found. Creating...${NC}"
    mkdir build
fi

cd build

# Clean up old benchmark results
echo -e "${YELLOW}Cleaning up old benchmark results...${NC}"
rm -f dtw_baseline_sequential.csv
rm -f dtw_benchmark_openmp.csv
rm -f dtw_benchmark_mpi_*.csv

# Step 1: Build and run pure sequential baseline
if $RUN_SEQUENTIAL; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  STEP 1: PURE SEQUENTIAL BASELINE    ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    echo -e "${BLUE}Building sequential benchmark (NO OpenMP/MPI)...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DUSE_OPENMP=OFF \
             -DUSE_MPI=OFF \
             -DUSE_CUDA=OFF \
             -DBUILD_BENCHMARKS=ON \
             -DBUILD_TESTS=OFF

    make benchmark_sequential_pure -j$(nproc)

    if [ $? -ne 0 ]; then
        echo -e "${RED}Sequential build failed!${NC}"
        exit 1
    fi

    echo -e "${BLUE}Running sequential baseline benchmark...${NC}"
    cd include/tests/performance
    ./benchmark_sequential_pure

    if [ $? -ne 0 ]; then
        echo -e "${RED}Sequential benchmark failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}Sequential baseline completed successfully${NC}"
    fi
    cd ../../..
fi

# Step 2: Build and run OpenMP benchmark
if $RUN_OPENMP; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    STEP 2: OPENMP BENCHMARK          ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    echo -e "${BLUE}Building OpenMP benchmark (OpenMP ONLY)...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DUSE_OPENMP=ON \
             -DUSE_MPI=OFF \
             -DUSE_CUDA=OFF \
             -DBUILD_BENCHMARKS=ON \
             -DBUILD_TESTS=OFF

    make benchmark_openmp_only -j$(nproc)

    if [ $? -ne 0 ]; then
        echo -e "${RED}OpenMP build failed!${NC}"
        exit 1
    fi

    echo -e "${BLUE}Running OpenMP benchmark...${NC}"
    cd include/tests/performance
    ./benchmark_openmp_only

    if [ $? -ne 0 ]; then
        echo -e "${RED}OpenMP benchmark failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}OpenMP benchmark completed successfully${NC}"
    fi
    cd ../../..
fi

# Step 3: Build and run MPI benchmark
if $RUN_MPI && command -v mpirun &> /dev/null; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}      STEP 3: MPI BENCHMARK           ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    echo -e "${BLUE}Building MPI benchmark (MPI ONLY)...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DUSE_OPENMP=OFF \
             -DUSE_MPI=ON \
             -DUSE_CUDA=OFF \
             -DBUILD_BENCHMARKS=ON \
             -DBUILD_TESTS=OFF

    make benchmark_mpi_only -j$(nproc)

    if [ $? -ne 0 ]; then
        echo -e "${RED}MPI build failed!${NC}"
        exit 1
    fi

    cd include/tests/performance

    # Run MPI benchmarks with different process counts
    for NPROCS in $MPI_PROCS; do
        echo ""
        echo -e "${BLUE}Running MPI benchmark with $NPROCS processes...${NC}"
        mpirun -np $NPROCS ./benchmark_mpi_only

        if [ $? -ne 0 ]; then
            echo -e "${RED}MPI benchmark with $NPROCS processes failed!${NC}"
        else
            echo -e "${GREEN}MPI benchmark with $NPROCS processes completed${NC}"
        fi
    done

    cd ../../..
elif ! $RUN_MPI; then
    echo -e "${YELLOW}MPI benchmarks skipped by user${NC}"
else
    echo -e "${YELLOW}MPI not found. Skipping MPI benchmarks.${NC}"
fi

# Step 4: Build and run CUDA benchmark (inserted after MPI, before plotting)
if $RUN_CUDA && command -v nvcc &> /dev/null; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}     STEP 4: CUDA BENCHMARK           ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    echo -e "${BLUE}Building CUDA benchmark (CUDA ONLY)...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DUSE_OPENMP=OFF \
             -DUSE_MPI=OFF \
             -DUSE_CUDA=ON \
             -DBUILD_BENCHMARKS=ON \
             -DBUILD_TESTS=OFF

    make benchmark_cuda_only -j$(nproc)
    if [ $? -ne 0 ]; then
        echo -e "${RED}CUDA build failed!${NC}"
        exit 1
    fi

    echo -e "${BLUE}Running CUDA benchmark...${NC}"
    cd include/tests/performance
    ./benchmark_cuda_only

    if [ $? -ne 0 ]; then
        echo -e "${RED}CUDA benchmark failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}CUDA benchmark completed successfully${NC}"
    fi
    cd ../../..
elif ! $RUN_CUDA; then
    echo -e "${YELLOW}CUDA benchmarks skipped by user${NC}"
else
    echo -e "${YELLOW}CUDA not found. Skipping CUDA benchmarks.${NC}"
fi

# Step 5: Generate plots
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   STEP 4: GENERATING PLOTS & REPORT  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

cd include/tests/performance

# Check if result files exist
if [ ! -f "dtw_baseline_sequential.csv" ]; then
    echo -e "${RED}Sequential baseline results not found!${NC}"
    echo -e "${RED}Run with --no-openmp --no-mpi to generate baseline first${NC}"
    exit 1
fi

echo -e "${BLUE}Generating plots and analysis...${NC}"
#python3 /mnt/c/Users/Francoo/CLionProjects/DTW-ACCELERATOR/include/tests/performance/plot_results_updated.py
PLOT_DIR="$ROOT_DIR/include/tests/performance"
PLOT_SCRIPT="$PLOT_DIR/plot_results_updated.py"
python3 "$PLOT_SCRIPT"
w
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   FAIR BENCHMARKS COMPLETE!          ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved in:"
    echo "  - dtw_baseline_sequential.csv (pure sequential baseline)"
    echo "  - dtw_benchmark_openmp.csv (OpenMP results)"
    echo "  - dtw_benchmark_mpi_*.csv (MPI results)"
    echo "  - dtw_benchmark_cuda.csv (CUDA results)"
    echo ""
    echo "Plots saved in benchmark_plots/:"
    echo "  - combined_*.png (comparison plots)"
    echo "  - scaling_efficiency.png"
    echo "  - cuda_performance.png (if CUDA available)"
    echo "  - benchmark_report.txt"
else
    echo -e "${RED}Plot generation failed!${NC}"
    exit 1
fi

# Return to original directory
cd ../../../..

echo ""
echo -e "${GREEN}All fair benchmarks completed successfully!${NC}"
echo -e "${YELLOW}Note: These results provide accurate speedup calculations${NC}"
echo -e "${YELLOW}by avoiding cross-contamination between parallel libraries.${NC}"