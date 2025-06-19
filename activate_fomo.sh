#!/bin/bash

# FOMO25 Environment Activation Script
# Usage: source activate_fomo.sh

# Handle Saturn Cloud's conda setup
if [ -f "/opt/saturncloud/etc/profile.d/conda.sh" ]; then
    source /opt/saturncloud/etc/profile.d/conda.sh
else
    # Fallback for other systems
    source $(conda info --base)/etc/profile.d/conda.sh
fi

# Activate the FOMO project environment
conda activate fomo_project

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "fomo_project" ]]; then
    echo "✅ Successfully activated fomo_project environment"
    echo "🐍 Python version: $(python --version)"
    echo "🔥 PyTorch version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")"
else
    echo "❌ Failed to activate fomo_project environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
fi