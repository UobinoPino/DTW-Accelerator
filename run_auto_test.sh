#!/bin/bash

# Script to build and run AutoStrategy validation test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    AutoStrategy Validation Test       ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check which backends to enable
ENABLE_OPENMP="OFF"
ENABLE_MPI="OFF"
ENABLE_CUDA="OFF"

# Check for OpenMP
if command -v g++ &> /dev/null && g++ -fopenmp -x c++ -E - < /dev/null &> /dev/null; then
    ENABLE_OPENMP="ON"
    echo -e "${GREEN}✓ OpenMP support detected${NC}"
else
    echo -e "${YELLOW}○ OpenMP not available${NC}"
fi

# Check for MPI
if command -v mpicxx &> /dev/null; then
    ENABLE_MPI="ON"
    echo -e "${GREEN}✓ MPI support detected${NC}"
else
    echo -e "${YELLOW}○ MPI not available${NC}"
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    ENABLE_CUDA="ON"
    echo -e "${GREEN}✓ CUDA support detected${NC}"
else
    echo -e "${YELLOW}○ CUDA not available${NC}"
fi

echo ""

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure CMake
echo -e "${BLUE}Configuring CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_OPENMP=${ENABLE_OPENMP} \
    -DUSE_MPI=${ENABLE_MPI} \
    -DUSE_CUDA=${ENABLE_CUDA} \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=OFF

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build the test
echo ""
echo -e "${BLUE}Building test_auto_strategy...${NC}"
make test_auto_strategy -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Run the test
echo ""
echo -e "${BLUE}Running AutoStrategy validation...${NC}"
echo ""

TEST_BINARY="${BUILD_DIR}/include/tests/test_auto_strategy"

if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${RED}Test binary not found at: $TEST_BINARY${NC}"
    exit 1
fi

# Run with MPI if available and enabled
if [ "$ENABLE_MPI" == "ON" ]; then
    echo -e "${YELLOW}Running with MPI (single process for testing)...${NC}"
    mpirun -np 1 "$TEST_BINARY"
else
    "$TEST_BINARY"
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   AutoStrategy Test Completed Successfully!   ${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}   AutoStrategy Test Failed!           ${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE