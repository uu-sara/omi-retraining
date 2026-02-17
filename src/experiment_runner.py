# -*- coding: utf-8 -*-
"""
Experiment runner for OMI detection model training and evaluation.

This module provides:
    - ExperimentRunner: Main class for running experiments with different configurations
    - Functions for training, validation, and testing with configurable settings
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add model directory to path for imports
MODEL_DIR = Path(__file__).parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from dataloader import BatchDataloader, OMIDataset
from metrics import EcgMetrics, AggregatedMetrics
from model import ECGModel, EnsembleECGModel
import utils

# Add configs to path
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
sys.path.insert(0, str(CONFIGS_DIR))

from base_config import ExperimentConfig, SIMPLIFIED_TO_DETAILED

logger = logging.getLogger(__name__)


class LossConfig:
    """Configure loss functions for mixed categorical/binary outcomes."""

    def __init__(
        self,
        outcomes_cat: List[str],
        outcomes_bin: List[str],
        outcome_columns: pd.Index,
        w_bin_cat_ratio: float = 1.0
    ):
        self.loss_fn_cat = nn.CrossEntropyLoss()
        self.loss_fn_bin = nn.BCEWithLogitsLoss()
        self.indx_cat = [outcome_columns.get_loc(col) for col in outcomes_cat]
        self.indx_bin = [outcome_columns.get_loc(col) for col in outcomes_bin]

        if len(outcomes_cat) == 0:
            self.w_bin, self.w_cat = 1.0, 0.0
        elif len(outcomes_bin) == 0:
            self.w_bin, self.w_cat = 0.0, 1.0
        else:
            self.w_bin = w_bin_cat_ratio / (w_bin_cat_ratio + 1)
            self.w_cat = 1 / (w_bin_cat_ratio + 1)

    def compute_loss(self, pred_logits, outcomes):
        loss_cat = self.loss_fn_cat(
            pred_logits[:, self.indx_cat], outcomes[:, self.indx_cat]
        ) if self.w_cat > 0 else None

        loss_bin = self.loss_fn_bin(
            pred_logits[:, self.indx_bin], outcomes[:, self.indx_bin]
        ) if self.w_bin > 0 else None

        total = torch.tensor(0.0)
        if loss_cat is not None:
            total = total + self.w_cat * loss_cat
        if loss_bin is not None:
            total = total + self.w_bin * loss_bin

        return total, loss_cat, loss_bin


class ExperimentRunner:
    """
    Main experiment runner for training and evaluating OMI detection models.

    Handles:
        - Directory setup and configuration saving
        - Training with ensemble support
        - Evaluation with both detailed and aggregated metrics
        - Result persistence

    Attributes:
        config: ExperimentConfig instance
        exp_dir: Path to experiment output directory
        device: Target device for computations
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the experiment runner.

        Args:
            config: ExperimentConfig with all experiment parameters
        """
        self.config = config
        self.exp_dir = Path(config.experiment_dir)
        self.device = self._setup_device()

        # Setup directories
        self._setup_directories()

        # Save configuration
        self._save_config()

        logger.info(f"Initialized experiment: {config.exp_name} ({config.exp_id})")

    def _setup_device(self) -> str:
        """Determine and return the target device."""
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")
        return device

    def _setup_directories(self) -> None:
        """Create experiment directory structure."""
        directories = [
            self.exp_dir,
            self.exp_dir / "checkpoints",
            self.exp_dir / "logs",
            self.exp_dir / "predictions",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created experiment directories at {self.exp_dir}")

    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_path = self.exp_dir / "config.json"
        self.config.save(str(config_path))
        logger.info(f"Saved configuration to {config_path}")

    def _get_args_namespace(self) -> Any:
        """Convert config to argparse.Namespace for compatibility."""
        import argparse

        # Get network architecture params
        net_filter_size, net_seq_length = utils.net_param_map(
            self.config.n_residual_block
        )

        args = argparse.Namespace(
            # Training params
            lr=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            patience=self.config.patience,
            min_lr=self.config.min_lr,
            lr_factor=self.config.lr_factor,
            weight_decay=self.config.weight_decay,
            dropout_rate=self.config.dropout_rate,
            optim_algo=self.config.optim_algo,
            seed=self.config.seed,

            # Model architecture
            n_leads=self.config.n_leads,
            seq_length=self.config.seq_length,
            n_residual_block=self.config.n_residual_block,
            net_filter_size=net_filter_size,
            net_seq_length=net_seq_length,
            agesex_dim=self.config.agesex_dim,
            kernel_size=self.config.kernel_size,
            activation_function=self.config.activation_function,

            # Ensemble
            n_ensembles=self.config.ensemble_size,

            # Outcomes
            outcomes_cat=self.config.outcomes_cat,
            outcomes_bin=self.config.outcomes_bin,
            col_outcome=self.config.col_outcome,
            n_outcomes=self.config.n_outcomes,
            w_bin_cat_ratio=self.config.w_bin_cat_ratio,

            # Data params
            hdf5=self.config.hdf5_path,
            txt=self.config.txt_path,
            split_col=self.config.split_col,
            age_col=self.config.age_col,
            male_col=self.config.male_col,
            age_mean=self.config.age_mean,
            age_sd=self.config.age_sd,

            # Feature flags
            include_age=self.config.include_age,
            include_sex=self.config.include_sex,

            # Output
            folder=str(self.exp_dir),
            device=self.device,

            # Category mapping
            use_simplified_categories=self.config.use_simplified_categories,
        )

        return args

    def prepare_data(
        self,
        test: bool = False,
        test_name: str = "test"
    ) -> Tuple[OMIDataset, Optional[BatchDataloader], Optional[BatchDataloader], Optional[BatchDataloader]]:
        """
        Prepare data loaders for training/validation or testing.

        Args:
            test: If True, prepare for testing only
            test_name: Name of test split

        Returns:
            Tuple of (dataset, train_loader, valid_loader, test_loader)
            Loaders are None if not applicable
        """
        args = self._get_args_namespace()

        dset = OMIDataset(
            path_to_h5=args.hdf5,
            path_to_txt=args.txt,
            col_outcome=args.col_outcome,
            split_col=args.split_col,
            age_col=args.age_col,
            male_col=args.male_col,
            age_mean=args.age_mean,
            age_sd=args.age_sd,
            test=test,
            test_name=test_name,
            use_simplified_categories=args.use_simplified_categories,
        )

        if test:
            test_loader = BatchDataloader(dset, args.batch_size, mask=dset.test)
            return dset, None, None, test_loader
        else:
            train_loader = BatchDataloader(dset, args.batch_size, mask=dset.train)
            valid_loader = BatchDataloader(dset, args.batch_size, mask=dset.valid)
            return dset, train_loader, valid_loader, None

    def train(self) -> Dict[str, Any]:
        """
        Train ensemble models according to experiment configuration.

        Returns:
            Dictionary with training results and paths to saved models
        """
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Training Experiment: {self.config.exp_name}")
        tqdm.write(f"{'='*60}\n")

        # Set random seed
        utils.seed_everything(self.config.seed, deterministic=True, warn_only=True)

        # Prepare data
        tqdm.write("Setting up data...", end="")
        dset, train_loader, valid_loader, _ = self.prepare_data()
        n_train, n_valid = sum(dset.train), sum(dset.valid)
        tqdm.write("done")
        tqdm.write(f"Found {n_train} training and {n_valid} validation records\n")

        # Save observed outcomes
        utils.save_dset(
            dset.outcomes[dset.train],
            str(self.exp_dir / "observed_data_train.csv")
        )
        utils.save_dset(
            dset.outcomes[dset.valid],
            str(self.exp_dir / "observed_data_valid.csv")
        )

        # Setup loss config
        args = self._get_args_namespace()
        loss_config = LossConfig(
            outcomes_cat=args.outcomes_cat,
            outcomes_bin=args.outcomes_bin,
            outcome_columns=dset.outcomes.columns,
            w_bin_cat_ratio=args.w_bin_cat_ratio
        )

        # Setup metrics
        metrics_calc = EcgMetrics(args.col_outcome)

        # Setup logger
        log = utils.Logger(str(self.exp_dir / "logs"))

        # Train ensemble members
        results = {
            "ensemble_size": self.config.ensemble_size,
            "models": [],
            "best_valid_loss": [],
        }

        for ensemble_nr in range(1, self.config.ensemble_size + 1):
            model_result = self._train_single_model(
                ensemble_nr=ensemble_nr,
                dset=dset,
                train_loader=train_loader,
                valid_loader=valid_loader,
                loss_config=loss_config,
                metrics_calc=metrics_calc,
                log=log,
                args=args
            )
            results["models"].append(model_result["model_path"])
            results["best_valid_loss"].append(model_result["best_valid_loss"])

        log.close()

        # Save training results summary
        results_path = self.exp_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        tqdm.write(f"\nTraining complete. Results saved to {results_path}")

        return results

    def _train_single_model(
        self,
        ensemble_nr: int,
        dset: OMIDataset,
        train_loader: BatchDataloader,
        valid_loader: BatchDataloader,
        loss_config: LossConfig,
        metrics_calc: EcgMetrics,
        log: utils.Logger,
        args: Any
    ) -> Dict[str, Any]:
        """Train a single ensemble member."""
        if args.n_ensembles > 1:
            tqdm.write(f"\nTraining ensemble member {ensemble_nr}/{args.n_ensembles}\n")

        n_valid = sum(dset.valid)
        true_outcomes_valid = dset.outcomes[dset.valid]

        # Create model
        model = ECGModel(args)
        model.to(args.device)

        # Setup optimizer
        if args.optim_algo.upper() == "SGD":
            optimizer = optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=args.patience,
            min_lr=args.lr_factor * args.min_lr,
            factor=args.lr_factor
        )

        # Setup paths
        if args.n_ensembles > 1:
            model_filename = f"model_{ensemble_nr}.pth"
            csv_filename = f"performance_metrics_{ensemble_nr}.csv"
        else:
            model_filename = "model.pth"
            csv_filename = "performance_metrics.csv"

        model_path = self.exp_dir / "checkpoints" / model_filename

        # Initialize logging
        log.init_tensorboardlog(suffix=str(ensemble_nr) if args.n_ensembles > 1 else "")
        log.init_csvlog()

        # Training loop
        best_loss = float("inf")

        for epoch_nr in range(1, args.epochs + 1):
            # Train epoch
            train_loss = self._train_epoch(
                epoch_nr, train_loader, optimizer, model, loss_config, args.device
            )

            # Validate epoch
            valid_loss, valid_pred = self._evaluate_epoch(
                epoch_nr, valid_loader, n_valid, args.n_outcomes,
                model, loss_config, args.device
            )

            # Compute metrics
            valid_metrics = metrics_calc.compute_metrics(valid_pred, true_outcomes_valid)

            # Check if best
            is_best = valid_loss["tot"] < best_loss
            if is_best:
                best_loss = valid_loss["tot"]
                torch.save({
                    "epoch": epoch_nr,
                    "ensemble": ensemble_nr,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "valid_loss": valid_loss["tot"],
                    "config": vars(args)
                }, model_path)

            # Check learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < args.min_lr:
                tqdm.write("Stopped: minimum learning rate reached")
                break

            # Log
            tqdm.write(
                f"Epoch {epoch_nr:3d}: "
                f"\033[91mTrain Loss\033[0m [tot={train_loss['tot']:.5f}] | "
                f"\033[91mValid Loss\033[0m [tot={valid_loss['tot']:.5f}] | "
                f"\033[91mLR\033[0m [{current_lr:.7f}] | "
                f"\033[91mBest\033[0m [{is_best}]"
            )

            log_data = {
                "ensemble": ensemble_nr,
                "epoch": epoch_nr,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in train_loss.items()},
                **{f"valid_{k}": v for k, v in valid_loss.items()},
                **valid_metrics
            }
            log.tensorboardlog_tofile(log_data, epoch_nr)
            log.csvlog_tofile(log_data, csv_filename)

            # Step scheduler
            scheduler.step(valid_loss["tot"])

        return {
            "model_path": str(model_path),
            "best_valid_loss": best_loss,
            "final_epoch": epoch_nr
        }

    def _train_epoch(
        self,
        epoch_nr: int,
        dataloader: BatchDataloader,
        optimizer: optim.Optimizer,
        model: ECGModel,
        loss_config: LossConfig,
        device: str
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss, total_loss_bin, total_loss_cat, n_samples = 0.0, 0.0, 0.0, 0

        pbar = tqdm(dataloader, desc=f"Training, epoch {epoch_nr:2d}", leave=False)

        for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
            ecgs = ecgs_cpu.to(device)
            outcomes = outcomes_cpu.to(device)
            age_sex = torch.stack([male_cpu, age_cpu], dim=1).to(device)

            optimizer.zero_grad()
            pred_logits = model((age_sex, ecgs))
            loss, loss_cat, loss_bin = loss_config.compute_loss(pred_logits, outcomes)
            loss.backward()
            optimizer.step()

            bs = ecgs_cpu.size(0)
            total_loss += loss.detach().cpu().item() * bs
            if loss_cat is not None:
                total_loss_cat += loss_cat.detach().cpu().item() * bs
            if loss_bin is not None:
                total_loss_bin += loss_bin.detach().cpu().item() * bs
            n_samples += bs

        pbar.close()

        metrics = {"tot": total_loss / n_samples}
        if loss_config.w_bin > 0 and loss_config.w_cat > 0:
            metrics["bin"] = total_loss_bin / n_samples
            metrics["cat"] = total_loss_cat / n_samples

        return metrics

    def _evaluate_epoch(
        self,
        epoch_nr: int,
        dataloader: BatchDataloader,
        n_records: int,
        n_outcomes: int,
        model: ECGModel,
        loss_config: LossConfig,
        device: str
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate for one epoch."""
        model.eval()
        total_loss, total_loss_bin, total_loss_cat, n_samples = 0.0, 0.0, 0.0, 0
        pred_prob = np.zeros((n_records, n_outcomes))
        batch_end = 0

        pbar = tqdm(dataloader, desc=f"Evaluation, epoch {epoch_nr:2d}", leave=False)

        with torch.no_grad():
            for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
                ecgs = ecgs_cpu.to(device)
                outcomes = outcomes_cpu.to(device)
                age_sex = torch.stack([male_cpu, age_cpu], dim=1).to(device)

                batch_start = batch_end
                bs = ecgs_cpu.size(0)

                pred_logits = model((age_sex, ecgs))
                loss, loss_cat, loss_bin = loss_config.compute_loss(pred_logits, outcomes)

                batch_end = min(batch_start + bs, n_records)

                # Apply activations
                if loss_config.indx_cat:
                    pred_prob[batch_start:batch_end, loss_config.indx_cat] = (
                        torch.softmax(pred_logits[:, loss_config.indx_cat], dim=1)
                        .cpu().numpy()
                    )
                if loss_config.indx_bin:
                    pred_prob[batch_start:batch_end, loss_config.indx_bin] = (
                        torch.sigmoid(pred_logits[:, loss_config.indx_bin])
                        .cpu().numpy()
                    )

                total_loss += loss.cpu().item() * bs
                if loss_cat is not None:
                    total_loss_cat += loss_cat.cpu().item() * bs
                if loss_bin is not None:
                    total_loss_bin += loss_bin.cpu().item() * bs
                n_samples += bs

        pbar.close()

        metrics = {"tot": total_loss / n_samples}
        if loss_config.w_bin > 0 and loss_config.w_cat > 0:
            metrics["bin"] = total_loss_bin / n_samples
            metrics["cat"] = total_loss_cat / n_samples

        return metrics, pred_prob

    def evaluate(
        self,
        test_name: str = "test",
        model_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate trained models on test set.

        Args:
            test_name: Name of test split in data
            model_dir: Directory containing trained models. If None, uses checkpoint dir.

        Returns:
            Dictionary with evaluation results
        """
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Evaluating Experiment: {self.config.exp_name}")
        tqdm.write(f"Test set: {test_name}")
        tqdm.write(f"{'='*60}\n")

        # Use checkpoint directory if not specified
        if model_dir is None:
            model_dir = str(self.exp_dir / "checkpoints")

        # Prepare data
        tqdm.write("Setting up data...", end="")
        dset, _, _, test_loader = self.prepare_data(test=True, test_name=test_name)
        n_test = sum(dset.test)
        true_outcomes_test = dset.outcomes[dset.test]
        tqdm.write("done")
        tqdm.write(f"Found {n_test} test records\n")

        # Save observed outcomes
        utils.save_dset(
            true_outcomes_test,
            str(self.exp_dir / f"observed_data_{test_name}.csv")
        )

        # Load model(s)
        args = self._get_args_namespace()

        # Setup loss config for index mapping
        loss_config = LossConfig(
            outcomes_cat=args.outcomes_cat,
            outcomes_bin=args.outcomes_bin,
            outcome_columns=dset.outcomes.columns,
            w_bin_cat_ratio=args.w_bin_cat_ratio
        )

        tqdm.write("Loading model...", end="")
        if self.config.ensemble_size > 1:
            model = EnsembleECGModel(
                config=args,
                model_dir=model_dir,
                aggregation_method=self.config.ensemble_aggregation,
                categorical_indices=loss_config.indx_cat,
                binary_indices=loss_config.indx_bin
            )
        else:
            model = ECGModel(args)
            model_path = os.path.join(model_dir, "model.pth")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            model.to(self.device)
            model.eval()
        tqdm.write("done\n")

        # Get predictions
        tqdm.write("Running predictions...")
        if self.config.ensemble_size > 1:
            test_pred, test_uncertainty = self._predict_ensemble(
                test_loader, n_test, args.n_outcomes, model
            )
        else:
            test_pred = self._predict_single(
                test_loader, n_test, args.n_outcomes, model, loss_config
            )
            test_uncertainty = None

        # Compute detailed metrics
        detailed_metrics = EcgMetrics(args.col_outcome)
        detailed_results = detailed_metrics.compute_metrics(test_pred, true_outcomes_test)

        # Compute aggregated metrics (OMI/NONOMI/CONTROL)
        if not self.config.use_simplified_categories:
            # Only aggregate if we trained on detailed categories
            agg_metrics = AggregatedMetrics(args.col_outcome)
            aggregated_results = agg_metrics.compute_metrics(test_pred, true_outcomes_test)
            classification_results = agg_metrics.compute_classification_metrics(
                test_pred, true_outcomes_test
            )
            cm, cm_labels = agg_metrics.get_confusion_matrix(test_pred, true_outcomes_test)
        else:
            # Already using simplified categories
            aggregated_results = detailed_results
            classification_results = {}
            cm, cm_labels = None, None

        # Combine results
        all_metrics = {
            **detailed_results,
            **{f"agg_{k}": v for k, v in aggregated_results.items()},
            **classification_results
        }

        # Save predictions
        pred_df = pd.DataFrame(
            data=test_pred,
            index=true_outcomes_test.index.tolist(),
            columns=[f"pr_{col}" for col in dset.outcomes.columns]
        )
        pred_df.index.name = true_outcomes_test.index.name
        pred_df.to_csv(self.exp_dir / "predictions" / f"predictions_{test_name}.csv")

        # Save uncertainty if available
        if test_uncertainty is not None:
            uncertainty_df = pd.DataFrame(
                data=test_uncertainty,
                index=true_outcomes_test.index.tolist(),
                columns=[f"std_{col}" for col in dset.outcomes.columns]
            )
            uncertainty_df.index.name = true_outcomes_test.index.name
            uncertainty_df.to_csv(
                self.exp_dir / "predictions" / f"uncertainty_{test_name}.csv"
            )

        # Save metrics
        metrics_path = self.exp_dir / f"metrics_{test_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        # Save confusion matrix
        if cm is not None:
            cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
            cm_df.to_csv(self.exp_dir / f"confusion_matrix_{test_name}.csv")

        # Print summary
        tqdm.write("\n" + detailed_metrics.get_summary(detailed_results))

        if not self.config.use_simplified_categories:
            agg_metrics_obj = AggregatedMetrics(args.col_outcome)
            tqdm.write("\n" + agg_metrics_obj.get_summary(aggregated_results))

        tqdm.write(f"\nResults saved to {metrics_path}")

        return {
            "detailed_metrics": detailed_results,
            "aggregated_metrics": aggregated_results,
            "classification_metrics": classification_results,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "confusion_matrix_labels": cm_labels
        }

    def _predict_ensemble(
        self,
        dataloader: BatchDataloader,
        n_records: int,
        n_outcomes: int,
        model: EnsembleECGModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from ensemble model."""
        model.eval()
        pred_prob = np.zeros((n_records, n_outcomes))
        pred_std = np.zeros((n_records, n_outcomes))
        batch_end = 0

        pbar = tqdm(dataloader, desc="Ensemble prediction", leave=False)

        with torch.no_grad():
            for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
                ecgs = ecgs_cpu.to(self.device)
                age_sex = torch.stack([male_cpu, age_cpu], dim=1).to(self.device)

                batch_start = batch_end
                bs = ecgs_cpu.size(0)

                mean_probs, std_probs = model.forward_with_uncertainty((age_sex, ecgs))

                batch_end = min(batch_start + bs, n_records)
                pred_prob[batch_start:batch_end, :] = mean_probs.cpu().numpy()
                pred_std[batch_start:batch_end, :] = std_probs.cpu().numpy()

        pbar.close()
        return pred_prob, pred_std

    def _predict_single(
        self,
        dataloader: BatchDataloader,
        n_records: int,
        n_outcomes: int,
        model: ECGModel,
        loss_config: LossConfig
    ) -> np.ndarray:
        """Get predictions from single model."""
        model.eval()
        pred_prob = np.zeros((n_records, n_outcomes))
        batch_end = 0

        pbar = tqdm(dataloader, desc="Single model prediction", leave=False)

        with torch.no_grad():
            for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
                ecgs = ecgs_cpu.to(self.device)
                age_sex = torch.stack([male_cpu, age_cpu], dim=1).to(self.device)

                batch_start = batch_end
                bs = ecgs_cpu.size(0)

                pred_logits = model((age_sex, ecgs))

                batch_end = min(batch_start + bs, n_records)

                # Apply activations
                if loss_config.indx_cat:
                    pred_prob[batch_start:batch_end, loss_config.indx_cat] = (
                        torch.softmax(pred_logits[:, loss_config.indx_cat], dim=1)
                        .cpu().numpy()
                    )
                if loss_config.indx_bin:
                    pred_prob[batch_start:batch_end, loss_config.indx_bin] = (
                        torch.sigmoid(pred_logits[:, loss_config.indx_bin])
                        .cpu().numpy()
                    )

        pbar.close()
        return pred_prob

    def run(self, test_name: str = "test") -> Dict[str, Any]:
        """
        Run full experiment: train and evaluate.

        Args:
            test_name: Name of test split for evaluation

        Returns:
            Dictionary with combined training and evaluation results
        """
        # Train
        train_results = self.train()

        # Evaluate
        eval_results = self.evaluate(test_name=test_name)

        # Combine results
        results = {
            "experiment": {
                "name": self.config.exp_name,
                "id": self.config.exp_id,
                "directory": str(self.exp_dir),
            },
            "training": train_results,
            "evaluation": eval_results,
        }

        # Save combined results
        results_path = self.exp_dir / "experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, default=str)

        return results
