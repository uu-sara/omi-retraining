#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gres=gpu:2
#SBATCH -t 03:00:00
#SBATCH -J test_ECG_retraining_workflow
#
# Run all OMI detection experiments
#
# Usage:
#   ./run_all_experiments.sh --hdf5 /path/to/data.hdf5 --txt /path/to/outcomes.txt
#   ./run_all_experiments.sh --hdf5 data_20pct.hdf5 --txt outcomes_20pct.txt --suffix subsample_20pct
#   ./run_all_experiments.sh --hdf5 data.hdf5 --txt outcomes.txt --output-dir results --epochs 50
#
# This script runs all 5 experiment configurations:
#   1. baseline - Full model with all features
#   2. simplified_categories - Training with OMI/NONOMI/CONTROL only
#   3. no_demographics - ECG only, no age/sex
#   4. single_model - No ensemble
#   5. probability_averaging - Different ensemble aggregation

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Singularity container for running Python
CONTAINER="/proj/nobackup/sens2020598/andreas/24.02-torch-py3.sif"

# Directory
SCRIPT_DIR="/proj/nobackup/sens2020598/andreas/omi-retraining"

# Default values
OUTPUT_DIR="experiments"
SUFFIX=""
EPOCHS=""
SKIP_COMPARE=false

# =============================================================================
# Argument parsing
# =============================================================================

print_usage() {
    echo "Usage: $0 --hdf5 <path> --txt <path> [options]"
    echo ""
    echo "Required arguments:"
    echo "  --hdf5 <path>        Path to HDF5 file with ECG data"
    echo "  --txt <path>         Path to text file with outcomes and splits"
    echo ""
    echo "Optional arguments:"
    echo "  --output-dir <path>  Output directory (default: experiments)"
    echo "  --suffix <string>    Suffix to append to experiment directory names"
    echo "                       (e.g., 'subsample_20pct' for subsampled data)"
    echo "  --epochs <num>       Override max epochs for all experiments"
    echo "  --skip-compare       Skip the comparison step at the end"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run all experiments with full data"
    echo "  $0 --hdf5 data.hdf5 --txt outcomes.txt"
    echo ""
    echo "  # Run with pre-subsampled 20% data"
    echo "  $0 --hdf5 data_subsample_20pct.hdf5 --txt outcomes_subsample_20pct.txt --suffix subsample_20pct"
    echo ""
    echo "  # Quick test with 1% data and 5 epochs"
    echo "  $0 --hdf5 data_subsample_1pct.hdf5 --txt outcomes_subsample_1pct.txt --suffix subsample_1pct --epochs 5"
}

# Parse arguments
HDF5_PATH=""
TXT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --hdf5)
            HDF5_PATH="$2"
            shift 2
            ;;
        --txt)
            TXT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-compare)
            SKIP_COMPARE=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$HDF5_PATH" ] || [ -z "$TXT_PATH" ]; then
    echo "Error: --hdf5 and --txt are required arguments"
    echo ""
    print_usage
    exit 1
fi

# Check files exist
if [ ! -f "$HDF5_PATH" ]; then
    echo "Error: HDF5 file not found: $HDF5_PATH"
    exit 1
fi

if [ ! -f "$TXT_PATH" ]; then
    echo "Error: Text file not found: $TXT_PATH"
    exit 1
fi

# Check Singularity container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Warning: Singularity container not found: $CONTAINER"
    echo "Falling back to direct Python execution"
    USE_SINGULARITY=false
else
    USE_SINGULARITY=true
fi

# =============================================================================
# Setup
# =============================================================================

# Build optional arguments string
EXTRA_ARGS=""
if [ -n "$SUFFIX" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --suffix $SUFFIX"
fi
if [ -n "$EPOCHS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
fi

# Function to run Python with or without Singularity
run_python() {
    if [ "$USE_SINGULARITY" = true ]; then
        singularity exec --nv "$CONTAINER" python "$@"
    else
        python "$@"
    fi
}

# =============================================================================
# Print configuration
# =============================================================================

echo "=============================================="
echo "OMI Detection Experiment Suite"
echo "=============================================="
echo "HDF5 file:        $HDF5_PATH"
echo "Text file:        $TXT_PATH"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$SUFFIX" ]; then
    echo "Suffix:           $SUFFIX"
fi
if [ -n "$EPOCHS" ]; then
    echo "Max epochs:       $EPOCHS"
fi
if [ "$USE_SINGULARITY" = true ]; then
    echo "Container:        $CONTAINER"
else
    echo "Container:        (not using Singularity)"
fi
echo ""

# =============================================================================
# Experiments to run
# =============================================================================

EXPERIMENTS=(
    "baseline"
    "simplified_categories"
    "no_demographics"
    "single_model"
    "probability_averaging"
)

# =============================================================================
# Run experiments
# =============================================================================

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running experiment: $exp"
    echo "=============================================="

    run_python "$SCRIPT_DIR/run_experiment.py" \
        --config "$exp" \
        --hdf5 "$HDF5_PATH" \
        --txt "$TXT_PATH" \
        --output-dir "$OUTPUT_DIR" \
        $EXTRA_ARGS

    echo "Experiment $exp completed!"
done

# =============================================================================
# Compare results
# =============================================================================

if [ "$SKIP_COMPARE" = false ]; then
    echo ""
    echo "=============================================="
    echo "Comparing experiment results"
    echo "=============================================="

    run_python "$SCRIPT_DIR/compare_experiments.py" \
        --experiments-dir "$OUTPUT_DIR" \
        --results-dir "$OUTPUT_DIR/results" \
        --outcomes-file "$TXT_PATH"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  Experiments: $OUTPUT_DIR/"
if [ "$SKIP_COMPARE" = false ]; then
    echo "  Comparisons: $OUTPUT_DIR/results/"
fi