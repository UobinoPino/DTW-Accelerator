#!/bin/bash
# Complete benchmark and plotting script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Parse command line arguments
USE_MPI=true
for arg in "$@"; do
    case $arg in
        --no-mpi)
            USE_MPI=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-mpi    Run without MPI even if installed"
            echo "  --help      Display this help message"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}DTW Comprehensive Benchmark Suite${NC}"
echo "======================================="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Build directory not found. Creating and configuring...${NC}"
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DUSE_OPENMP=ON \
             -DUSE_MPI=OFF \
             -DUSE_CUDA=ON \
             -DBUILD_BENCHMARKS=ON
else
    cd build
fi

# Build the benchmark
echo -e "${GREEN}Building benchmark...${NC}"
make benchmark_comprehensive -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Navigate to benchmark directory
cd include/tests/performance

# Run benchmark
echo -e "${GREEN}Running benchmarks...${NC}"
if command -v mpirun &> /dev/null && $USE_MPI; then
    echo "Running with MPI (4 processes)..."
    mpirun -np 4 ./benchmark_comprehensive
else
    if command -v mpirun &> /dev/null && ! $USE_MPI; then
        echo "MPI detected but disabled by user. Running without MPI..."
    else
        echo "Running without MPI..."
    fi
    ./benchmark_comprehensive
fi

# Check if CSV was generated
if [ ! -f "dtw_benchmark_results.csv" ]; then
    echo -e "${RED}Benchmark CSV not generated!${NC}"
    exit 1
fi

# Generate plots
echo -e "${GREEN}Generating plots...${NC}"
python3 /mnt/c/Users/Francoo/CLionProjects/DTW-ACCELERATOR/include/tests/performance/plot_results.py

echo -e "${GREEN}Benchmark complete! Check 'benchmark_plots' directory for results.${NC}"