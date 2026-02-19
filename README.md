# OMI Retraining

This repository contains code for retraining and evaluating the OMI (Occlusion Myocardial Infarction) detection model with various new configurations.

## Original Model

The neural network model architecture and core training code in the `model/` directory is based on the work of **Stefan Gustafsson** from Uppsala University. The original implementation is available at:

**https://github.com/stefan-gustafsson-work/omi**

## Overview

This repository adds:
- Experiment configurations
- Subsampling tools (to use less data for training, for speed improvement)
- GPU-optimized training with PyTorch DataLoader
- Experiment comparison and visualization of results

## Experiment Configurations

| Config | Description |
|--------|-------------|
| `baseline` | Full model with all features and 5-model ensemble |
| `simplified_categories` | Training with OMI/NONOMI/CONTROL only |
| `no_demographics` | ECG only, no age/sex features |
| `single_model` | No ensemble (single model) |
| `probability_averaging` | Ensemble with probability averaging instead of logit averaging |

## Usage

### Running a Single Experiment

```bash
python run_experiment.py --config baseline --hdf5 data.hdf5 --txt outcomes.txt --output-dir experiments
```

With a specific test split name:
```bash
python run_experiment.py --config baseline --hdf5 data.hdf5 --txt outcomes.txt --output-dir experiments --test-name test_rand
```

### Comparing Experiments

Compare all experiments:
```bash
python compare_experiments.py --test-name test_rand --outcomes-file /path/to/outcomes.txt --results-dir experiments/results
```

Compare specific experiments:
```bash
python compare_experiments.py --test-name test_rand --outcomes-file /path/to/outcomes.txt --include baseline single_model
```

### Data Subsampling

Create subsampled datasets (preserves rare classes, subsamples majority class):
```bash
python subsample_hdf5.py --hdf5 data.hdf5 --txt outcomes.txt --output-dir ./subsets --percentages 1 20
```

Extract specific splits to separate files:
```bash
python extract_splits.py --hdf5 data.hdf5 --txt outcomes.txt --output-dir ./test_sets --splits test_rand test_temp
```

Verify class distributions:
```bash
python verify_subsampling.py --original outcomes.txt --subsampled outcomes_20pct.txt
```

Count samples by split and outcome:
```bash
python count_by_split.py outcomes.txt
```

## Requirements

- Python 3.8+
- PyTorch
- h5py
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for plotting)
- seaborn (optional, for plotting)

## License

The model code in `model/` is based on Stefan Gustafsson's original implementation. Please refer to the original repository for licensing information: https://github.com/stefan-gustafsson-work/omi
