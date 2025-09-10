#!/bin/bash

# Script to run unit tests suite

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BIN_DIR="/mnt/c/Users/Francoo/CLionProjects/DTW-ACCELERATOR/build/include/tests"
TESTS=( "test_core" "test_distance_metrics" "test_strategies" "test_constraints" )

echo "======================================"
echo "DTW Accelerator Unit Test Suite"
echo "Directory dei test: $BIN_DIR"
echo "======================================"
echo ""

if [ ! -d "$BIN_DIR" ]; then
    echo -e "${RED}Error: Directory dei test non trovata: ${BIN_DIR}${NC}"
    exit 1
fi

cd "$BIN_DIR" || exit 1

echo -e "${BLUE}Running unit tests...${NC}"
echo ""

EXIT_CODE=0

for test in "${TESTS[@]}"; do
    echo "----------------------------------------"
    echo "Running $test..."
    echo "----------------------------------------"

    if [ ! -f "./$test" ]; then
        echo -e "${YELLOW}Skipping $test: file non trovato${NC}"
        EXIT_CODE=2
        echo ""
        continue
    fi

    if [ ! -x "./$test" ]; then
        chmod +x "./$test" 2>/dev/null || true
    fi

    ./"$test"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo -e "${RED}✗ $test failed (exit $rc)${NC}"
        EXIT_CODE=$rc
    else
        echo -e "${GREEN}✓ $test passed${NC}"
    fi
    echo ""
done

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
else
    echo -e "${RED}✗ Some tests failed or were skipped. Exit code: ${EXIT_CODE}${NC}"
fi

exit $EXIT_CODE
#c