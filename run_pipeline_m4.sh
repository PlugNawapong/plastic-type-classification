#!/bin/bash

# MacBook Air M4 optimized pipeline runner
# Fixes OpenMP library conflicts on macOS

# Set environment variable to handle OpenMP library conflicts
export KMP_DUPLICATE_LIB_OK=TRUE

# Optional: Optimize for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the pipeline with all arguments passed through
python run_pipeline_config.py "$@"
