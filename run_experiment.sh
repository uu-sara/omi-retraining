#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gres=gpu:2
#SBATCH -t 03:00:00
#SBATCH -J omi_experiment
#
# Run a single OMI detection experiment
#
# Usage:
#   ./run_experiment.sh --config baseline --hdf5 /path/to/data.hdf5 --txt /path/to/outcomes.txt
#   ./run_experiment.sh --config baseline --hdf5 data_20pct.hdf5 --txt outcomes_20pct.txt --suffix subsample_20pct
#
# Available experiment configurations:
#   - baseline                 Full model with all features
#   - simplified_categories    Training with OMI/NONOMI/CONTROL only
#   - no_demographics          ECG only, no age/sex
#   - single_model             No ensemble
#   - probability_averaging    Different ensemble aggregation

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
TEST_NAME=""
TRAIN_ONLY=false
EVAL_ONLY=false

# =============================================================================
# Argument parsing
# =============================================================================

print_usage() {
    echo "Usage: $0 --config <name> --hdf5 <path> --txt <path> [options]"
    echo ""
    echo "Required arguments:"
    echo "  --config <name>      Experiment configuration to run"
    echo "                       Options: baseline, simplified_categories, no_demographics,"
    echo "                                single_model, probability_averaging"
    echo "  --hdf5 <path>        Path to HDF5 file with ECG data"
    echo "  --txt <path>         Path to text file with outcomes and splits"
    echo ""
    echo "Optional arguments:"
    echo "  --output-dir <path>  Output directory (default: experiments)"
    echo "  --suffix <string>    Suffix to append to experiment directory name"
    echo "                       (e.g., 'subsample_20pct' for subsampled data)"
    echo "  --epochs <num>       Override max epochs"
    echo "  --test-name <name>   Name of test split (default: 'test')"
    echo "                       Use 'test_rand' or 'test_temp' for your data"
    echo "  --train-only         Only train, skip evaluation"
    echo "  --eval-only          Only evaluate (requires pre-trained models)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run baseline experiment with full data"
    echo "  $0 --config baseline --hdf5 data.hdf5 --txt outcomes.txt"
    echo ""
    echo "  # Run with pre-subsampled 20% data"
    echo "  $0 --config baseline --hdf5 data_subsample_20pct.hdf5 --txt outcomes_subsample_20pct.txt --suffix subsample_20pct"
    echo ""
    echo "  # Quick test with 1% data and 5 epochs"
    echo "  $0 --config baseline --hdf5 data_subsample_1pct.hdf5 --txt outcomes_subsample_1pct.txt --suffix subsample_1pct --epochs 5"
}

# Parse arguments
CONFIG=""
HDF5_PATH=""
TXT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --hdf5)
            HDF5_PATH="$2"
            shift 2
            ;;
        --txt)
            TXT_PATH="$2"
            shift 2
            ;;
        --output-dir|-o)
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
        --test-name)
            TEST_NAME="$2"
            shift 2
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
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
if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    echo ""
    print_usage
    exit 1
fi

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
if [ -n "$TEST_NAME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --test-name $TEST_NAME"
fi
if [ "$TRAIN_ONLY" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --train-only"
fi
if [ "$EVAL_ONLY" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --eval-only"
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
echo "OMI Detection Experiment"
echo "=============================================="
echo "Config:           $CONFIG"
echo "HDF5 file:        $HDF5_PATH"
echo "Text file:        $TXT_PATH"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$SUFFIX" ]; then
    echo "Suffix:           $SUFFIX"
fi
if [ -n "$EPOCHS" ]; then
    echo "Max epochs:       $EPOCHS"
fi
if [ "$TRAIN_ONLY" = true ]; then
    echo "Mode:             Train only"
elif [ "$EVAL_ONLY" = true ]; then
    echo "Mode:             Eval only"
else
    echo "Mode:             Train + Eval"
fi
if [ "$USE_SINGULARITY" = true ]; then
    echo "Container:        $CONTAINER"
else
    echo "Container:        (not using Singularity)"
fi
echo ""

# =============================================================================
# Run experiment
# =============================================================================

echo "=============================================="
echo "Running experiment: $CONFIG"
echo "=============================================="

run_python "$SCRIPT_DIR/run_experiment.py" \
    --config "$CONFIG" \
    --hdf5 "$HDF5_PATH" \
    --txt "$TXT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
