#!/bin/bash

# FOMO25 Project Environment Setup Script
# This script automatically sets up the correct Python environment for the FOMO25 project

set -e  # Exit on any error

echo "🚀 Setting up FOMO25 project environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first."
    exit 1
fi

# Environment name
ENV_NAME="fomo_project"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "✅ Using existing environment."
        # Handle Saturn Cloud's conda setup
        if [ -f "/opt/saturncloud/etc/profile.d/conda.sh" ]; then
            source /opt/saturncloud/etc/profile.d/conda.sh
        else
            source $(conda info --base)/etc/profile.d/conda.sh
        fi
        conda activate ${ENV_NAME}
        echo "Environment activated. You can now use: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create conda environment with Python 3.11.8
echo "🐍 Creating conda environment with Python 3.11.8..."
conda create -n ${ENV_NAME} python=3.11.8 -y

# Activate environment
echo "🔄 Activating environment..."
# Handle Saturn Cloud's conda setup
if [ -f "/opt/saturncloud/etc/profile.d/conda.sh" ]; then
    source /opt/saturncloud/etc/profile.d/conda.sh
else
    source $(conda info --base)/etc/profile.d/conda.sh
fi
conda activate ${ENV_NAME}

# Verify Python version
echo "✅ Python version: $(python --version)"

# Install PyTorch with CUDA support (compatible with torch<2.3.0 requirement)
echo "🔥 Installing PyTorch 2.2.2 with CUDA support..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
echo "✅ PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo "✅ CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"

# Install the project in development mode with dev and test dependencies
echo "📦 Installing FOMO25 project dependencies (including dev and test)..."
pip install -e ".[dev,test]"

echo ""
echo "🎉 Setup complete! Your environment is ready."
echo ""
echo "To activate the environment, run:"
echo "   source /opt/saturncloud/etc/profile.d/conda.sh"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "Or use the helper script:"
echo "   source activate_fomo.sh"
echo ""
echo "To test the setup:"
echo "   source activate_fomo.sh"
echo "   python -c \"import yucca; print('Yucca works')\""
echo ""
echo "To run your training script:"
echo "   source activate_fomo.sh"
echo "   python src/pretrain.py [your arguments]"
echo ""