#!/bin/bash

# ============================================================================
# Python Virtual Environment Setup for DTW Accelerator
# Minimal setup with essential dependencies only
# ============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="dtw_env"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DTW Accelerator - Python Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check Python installation
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ Error: Python not found! Please install Python 3.6+${NC}"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"
}

# Create virtual environment
create_venv() {
    echo -e "\n${BLUE}Creating virtual environment...${NC}"

    # Remove old venv if exists
    if [ -d "$VENV_NAME" ]; then
        echo -e "${YELLOW}  Removing existing environment...${NC}"
        rm -rf "$VENV_NAME"
    fi

    # Create new virtual environment
    $PYTHON_CMD -m venv $VENV_NAME

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create virtual environment!${NC}"
        echo -e "${YELLOW}Try: sudo apt-get install python3-venv${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Virtual environment created${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "\n${BLUE}Installing dependencies...${NC}"

    # Activate virtual environment
    source $VENV_NAME/bin/activate

    # Upgrade pip
    pip install --upgrade pip -q

    # Install essential packages
    echo -e "  Installing numpy..."
    pip install numpy -q
    echo -e "${GREEN}  ✓ numpy installed${NC}"

    echo -e "  Installing pandas..."
    pip install pandas -q
    echo -e "${GREEN}  ✓ pandas installed${NC}"

    echo -e "  Installing matplotlib..."
    pip install matplotlib -q
    echo -e "${GREEN}  ✓ matplotlib installed${NC}"

    # Optional but useful for better plots
    echo -e "  Installing seaborn (for better plot styles)..."
    pip install seaborn -q
    echo -e "${GREEN}  ✓ seaborn installed${NC}"
}

# Verify installation
verify_installation() {
    echo -e "\n${BLUE}Verifying installation...${NC}"

    python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "  ✗ NumPy"
    python -c "import pandas; print(f'  ✓ Pandas {pandas.__version__}')" 2>/dev/null || echo "  ✗ Pandas"
    python -c "import matplotlib; print(f'  ✓ Matplotlib {matplotlib.__version__}')" 2>/dev/null || echo "  ✗ Matplotlib"
    python -c "import seaborn; print(f'  ✓ Seaborn {seaborn.__version__}')" 2>/dev/null || echo "  ✗ Seaborn"
}

# Main execution
main() {
    check_python
    create_venv
    install_dependencies
    verify_installation

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}       Setup Complete! 🎉${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    echo -e "${YELLOW}To activate the environment:${NC}"
    echo -e "  source dtw_env/bin/activate\n"

    echo -e "${YELLOW}To run benchmark visualization:${NC}"
    echo -e "  cd build/include/tests/performance"
    echo -e "  python plot_results.py\n"

    echo -e "${YELLOW}To deactivate when done:${NC}"
    echo -e "  deactivate\n"
}

# Run main function
main