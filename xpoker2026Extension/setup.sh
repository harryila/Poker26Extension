#!/bin/bash
# Setup script for Poker Environment
# Usage: ./setup.sh [--with-gpu] [--hf-token TOKEN]
#
# Options:
#   --with-gpu     Install PyTorch with CUDA support
#   --hf-token     HuggingFace token for model access

set -e

# Parse arguments
WITH_GPU=false
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-gpu)
            WITH_GPU=true
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Poker Environment Setup ==="

# Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD=$cmd
            echo "Found Python $version at $(which $cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11+ is required but not found."
    echo ""
    echo "Install options:"
    echo "  macOS:   brew install python@3.11"
    echo "  Ubuntu:  sudo apt install python3.11 python3.11-venv"
    echo "  pyenv:   pyenv install 3.11"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch with appropriate backend
if [ "$WITH_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate huggingface_hub
fi

# Install remaining requirements
pip install -r requirements.txt

# HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "HuggingFace login successful!"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run a basic experiment (no LLM):"
echo "  python run_experiment.py --agent random --hands 100 --seed 42 -v"
echo ""
echo "To run with LLM agent (requires GPU + HF access):"
echo "  python run_experiment.py --agent hf --opponent call --hands 5 --elicit-beliefs -v"
echo ""
echo "To run tests:"
echo "  pytest poker_env/tests/ -v"
echo ""

# Check GPU availability
if [ "$WITH_GPU" = true ]; then
    echo "=== GPU Check ==="
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null || echo "PyTorch not installed or CUDA not available"
    echo ""
fi

# HuggingFace login reminder
if [ -z "$HF_TOKEN" ]; then
    echo "=== HuggingFace Setup (for LLM agent) ==="
    echo "To use the HF agent, you need to:"
    echo "  1. Create account at https://huggingface.co"
    echo "  2. Request access to meta-llama/Llama-3.1-8B-Instruct"
    echo "  3. Login with: python -c \"from huggingface_hub import login; login()\""
    echo ""
    echo "Or re-run setup with token:"
    echo "  ./setup.sh --with-gpu --hf-token YOUR_TOKEN"
    echo ""
fi
