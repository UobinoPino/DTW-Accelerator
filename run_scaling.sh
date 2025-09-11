#!/bin/bash
# DTW Scaling Benchmark Script
# Runs OpenMP (1-8 threads) and MPI (1-8 processes) scaling benchmarks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_OPENMP=true
RUN_MPI=true

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --no-openmp)
            RUN_OPENMP=false
            shift
            ;;
        --no-mpi)
            RUN_MPI=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-openmp  Skip OpenMP scaling benchmarks"
            echo "  --no-mpi     Skip MPI scaling benchmarks"
            echo "  --help       Display this help message"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     DTW SCALING BENCHMARK SUITE      ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Build directory not found. Creating and configuring...${NC}"
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DUSE_OPENMP=ON \
             -DUSE_MPI=ON \
             -DUSE_CUDA=OFF \
             -DBUILD_BENCHMARKS=ON
else
    cd build
fi

# Build the scaling benchmark
echo -e "${GREEN}Building scaling benchmark...${NC}"
make benchmark_scaling -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Navigate to benchmark directory
cd include/tests/performance

# Clean up old results
echo -e "${YELLOW}Cleaning up old scaling results...${NC}"
rm -f dtw_scaling_results*.csv

# Run OpenMP scaling benchmarks (sequential process handles OpenMP internally)
if $RUN_OPENMP; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  RUNNING OPENMP SCALING (1-8 threads) ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    ./benchmark_scaling

    if [ $? -ne 0 ]; then
        echo -e "${RED}OpenMP scaling benchmark failed!${NC}"
    else
        echo -e "${GREEN}OpenMP scaling benchmark completed successfully${NC}"
    fi
fi

# Run MPI scaling benchmarks (requires multiple mpirun calls)
if $RUN_MPI && command -v mpirun &> /dev/null; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    RUNNING MPI SCALING (1-8 processes) ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Run for different process counts
    for NPROCS in 1 2 4 8; do
        echo -e "${BLUE}Running MPI with $NPROCS processes...${NC}"
        mpirun -np $NPROCS ./benchmark_scaling

        if [ $? -ne 0 ]; then
            echo -e "${RED}MPI benchmark with $NPROCS processes failed!${NC}"
        else
            echo -e "${GREEN}MPI benchmark with $NPROCS processes completed${NC}"
        fi
        echo ""
    done
elif ! $RUN_MPI; then
    echo -e "${YELLOW}MPI scaling benchmarks skipped by user${NC}"
else
    echo -e "${YELLOW}MPI not found. Skipping MPI scaling benchmarks.${NC}"
fi

# Check if any CSV files were generated
if ! ls dtw_scaling_results*.csv 1> /dev/null 2>&1; then
    echo -e "${RED}No scaling results generated!${NC}"
    exit 1
fi

# Generate plots
echo ""
echo -e "${GREEN}Generating scaling plots...${NC}"
python3 /mnt/c/Users/Francoo/CLionProjects/DTW-ACCELERATOR/include/tests/performance/plot_results.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   SCALING BENCHMARKS COMPLETE!        ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved in:"
    echo "  - dtw_scaling_results*.csv (raw data)"
    echo "  - benchmark_plots/openmp_scaling.png"
    echo "  - benchmark_plots/mpi_scaling.png"
    echo "  - benchmark_plots/benchmark_report.txt"
else
    echo -e "${RED}Plot generation failed!${NC}"
    exit 1
fi

# Return to original directory
cd ../../../..

echo ""
echo -e "${GREEN}All scaling benchmarks completed successfully!${NC}"