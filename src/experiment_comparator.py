# -*- coding: utf-8 -*-
"""
Experiment comparison and visualization tools.

This module provides:
    - ExperimentComparator: Load and compare results across experiments
    - Visualization functions for offline plotting
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

# Optional plotting imports - handle gracefully if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for offline plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)

# Mapping from detailed categories to simplified categories
CATEGORY_MAPPING = {
    "control_nomyoperi": "control",
    "control_myoperi": "control",
    "mi_nstemi_nonomi": "nonomi",
    "mi_stemi_nonomi": "nonomi",
    "mi_nstemi_omi_lmca_lad": "omi",
    "mi_nstemi_omi_lcx": "omi",
    "mi_nstemi_omi_rca": "omi",
    "mi_stemi_omi_lmca_lad": "omi",
    "mi_stemi_omi_lcx": "omi",
    "mi_stemi_omi_rca": "omi",
}

# Detailed diagnosis columns from outcomes file
DETAILED_DIAGNOSIS_COLS = [
    "control_nomyoperi",
    "control_myoperi",
    "mi_nstemi_nonomi",
    "mi_stemi_nonomi",
    "mi_nstemi_omi_lmca_lad",
    "mi_nstemi_omi_lcx",
    "mi_nstemi_omi_rca",
    "mi_stemi_omi_lmca_lad",
    "mi_stemi_omi_lcx",
    "mi_stemi_omi_rca",
]

# Mapping from detailed columns to 5-group labels
FIVE_GROUP_MAPPING = {
    "control_nomyoperi": "CONTROL",
    "control_myoperi": "CONTROL",
    "mi_nstemi_nonomi": "NSTEMI-NONOMI",
    "mi_stemi_nonomi": "STEMI-NONOMI",
    "mi_nstemi_omi_lmca_lad": "NSTEMI-OMI",
    "mi_nstemi_omi_lcx": "NSTEMI-OMI",
    "mi_nstemi_omi_rca": "NSTEMI-OMI",
    "mi_stemi_omi_lmca_lad": "STEMI-OMI",
    "mi_stemi_omi_lcx": "STEMI-OMI",
    "mi_stemi_omi_rca": "STEMI-OMI",
}

# OMI detailed columns (for subgroup filtering)
OMI_COLS = [
    "mi_nstemi_omi_lmca_lad", "mi_nstemi_omi_lcx", "mi_nstemi_omi_rca",
    "mi_stemi_omi_lmca_lad", "mi_stemi_omi_lcx", "mi_stemi_omi_rca",
]
NONOMI_COLS = ["mi_nstemi_nonomi", "mi_stemi_nonomi"]
NSTEMI_OMI_COLS = ["mi_nstemi_omi_lmca_lad", "mi_nstemi_omi_lcx", "mi_nstemi_omi_rca"]

THRESHOLDS = [0.05, 0.01, 0.005]

FIVE_GROUP_ORDER = ["NSTEMI-OMI", "STEMI-OMI", "NSTEMI-NONOMI", "STEMI-NONOMI", "CONTROL"]
THREE_GROUP_ORDER = ["OMI", "NONOMI", "CONTROL"]


class ExperimentComparator:
    """
    Compare results across multiple experiments.

    Loads experiment results from directory structure and provides
    methods for creating comparison tables and visualizations.
    """

    def __init__(
        self,
        experiments_dir: str = "experiments",
        results_dir: str = "results",
        outcomes_file: str = "example_data/test_outcomes.txt",
    ) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir)
        self.outcomes_file = Path(outcomes_file)
        self.experiments: Dict[str, Dict[str, Any]] = {}

        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized comparator with experiments_dir={experiments_dir}")

    def _load_demographics(self) -> pd.DataFrame:
        """Load demographics (age, male) and detailed diagnosis columns from outcomes file."""
        df = pd.read_csv(self.outcomes_file, sep="\t")
        keep_cols = ["id_record", "age", "male"] + [
            c for c in DETAILED_DIAGNOSIS_COLS if c in df.columns
        ]
        return df[keep_cols]

    def load_experiment(self, exp_path: Path) -> Optional[Dict[str, Any]]:
        """Load results from a single experiment directory."""
        try:
            config_path = exp_path / "config.json"
            if not config_path.exists():
                logger.warning(f"No config.json found in {exp_path}")
                return None

            with open(config_path) as f:
                config = json.load(f)

            # Load metrics
            metrics = {}
            for metrics_file in ["metrics_test.json", "metrics.json"]:
                metrics_path = exp_path / metrics_file
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    break

            # Load training results
            training_results = {}
            training_path = exp_path / "training_results.json"
            if training_path.exists():
                with open(training_path) as f:
                    training_results = json.load(f)

            # Load confusion matrix
            confusion_matrix = None
            cm_labels = None
            for cm_file in ["confusion_matrix_test.csv", "confusion_matrix.csv"]:
                cm_path = exp_path / cm_file
                if cm_path.exists():
                    cm_df = pd.read_csv(cm_path, index_col=0)
                    confusion_matrix = cm_df.values
                    cm_labels = cm_df.columns.tolist()
                    break

            # Load predictions
            predictions_df = None
            pred_path = exp_path / "predictions" / "predictions_test.csv"
            if pred_path.exists():
                predictions_df = pd.read_csv(pred_path)

            # Load observed data
            observed_df = None
            obs_path = exp_path / "observed_data_test.csv"
            if obs_path.exists():
                observed_df = pd.read_csv(obs_path)

            return {
                "path": str(exp_path),
                "config": config,
                "metrics": metrics,
                "training": training_results,
                "confusion_matrix": confusion_matrix,
                "cm_labels": cm_labels,
                "predictions_df": predictions_df,
                "observed_df": observed_df,
            }

        except Exception as e:
            logger.warning(f"Failed to load experiment from {exp_path}: {e}")
            return None

    def load_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load results from all experiments in experiments directory."""
        if not self.experiments_dir.exists():
            logger.warning(f"Experiments directory not found: {self.experiments_dir}")
            return {}

        self.experiments = {}
        for exp_path in sorted(self.experiments_dir.iterdir()):
            if not exp_path.is_dir():
                continue
            # Skip the results directory
            if exp_path.name == "results":
                continue

            result = self.load_experiment(exp_path)
            if result is not None:
                exp_name = result["config"].get("exp_name", exp_path.name)
                self.experiments[exp_name] = result
                logger.info(f"Loaded experiment: {exp_name}")

        logger.info(f"Loaded {len(self.experiments)} experiments")
        return self.experiments

    @staticmethod
    def _get_simplified_probabilities(predictions_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Get simplified (omi/nonomi/control) probabilities from predictions.

        For detailed experiments: sum probability columns mapping to each simplified category.
        For simplified experiments: columns already are pr_omi, pr_nonomi, pr_control.
        """
        result = pd.DataFrame({"id_record": predictions_df["id_record"]})

        if config.get("use_simplified_categories", False):
            result["pr_omi"] = predictions_df["pr_omi"]
            result["pr_nonomi"] = predictions_df["pr_nonomi"]
            result["pr_control"] = predictions_df["pr_control"]
        else:
            # Sum detailed columns into simplified categories
            simplified_sums = {"pr_omi": 0.0, "pr_nonomi": 0.0, "pr_control": 0.0}
            for detailed_col, simplified_cat in CATEGORY_MAPPING.items():
                pr_col = f"pr_{detailed_col}"
                if pr_col in predictions_df.columns:
                    simplified_sums[f"pr_{simplified_cat}"] = (
                        simplified_sums[f"pr_{simplified_cat}"] + predictions_df[pr_col]
                    )
            result["pr_omi"] = simplified_sums["pr_omi"]
            result["pr_nonomi"] = simplified_sums["pr_nonomi"]
            result["pr_control"] = simplified_sums["pr_control"]

        return result

    @staticmethod
    def _get_5group_true_labels(demographics_df: pd.DataFrame) -> pd.Series:
        """
        Assign each sample to one of 5 groups based on detailed diagnosis columns.

        Returns a Series with values: NSTEMI-OMI, STEMI-OMI, NSTEMI-NONOMI, STEMI-NONOMI, CONTROL
        """
        labels = pd.Series("CONTROL", index=demographics_df.index)
        for col, group in FIVE_GROUP_MAPPING.items():
            if col in demographics_df.columns:
                mask = demographics_df[col] == 1
                labels[mask] = group
        return labels

    @staticmethod
    def _classify_at_threshold(pr_omi: np.ndarray, pr_nonomi: np.ndarray, threshold: float) -> np.ndarray:
        """
        Classify predictions at a given threshold.

        If pr_omi >= threshold -> OMI
        Elif pr_nonomi >= threshold -> NONOMI
        Else -> CONTROL
        """
        predicted = np.full(len(pr_omi), "CONTROL", dtype=object)
        predicted[pr_omi >= threshold] = "OMI"
        # Only classify as NONOMI if not already OMI
        nonomi_mask = (pr_nonomi >= threshold) & (pr_omi < threshold)
        predicted[nonomi_mask] = "NONOMI"
        return predicted

    def _compute_subgroup_masks(self, merged_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return dict of subgroup_name -> boolean mask."""
        masks = {}

        # OMI subgroup
        omi_cols_present = [c for c in OMI_COLS if c in merged_df.columns]
        if omi_cols_present:
            masks["OMI"] = merged_df[omi_cols_present].max(axis=1) == 1

        # NONOMI subgroup
        nonomi_cols_present = [c for c in NONOMI_COLS if c in merged_df.columns]
        if nonomi_cols_present:
            masks["NONOMI"] = merged_df[nonomi_cols_present].max(axis=1) == 1

        # NSTEMI-OMI subgroup
        nstemi_omi_present = [c for c in NSTEMI_OMI_COLS if c in merged_df.columns]
        if nstemi_omi_present:
            masks["NSTEMI-OMI"] = merged_df[nstemi_omi_present].max(axis=1) == 1

        # Gender
        if "male" in merged_df.columns:
            masks["MEN"] = merged_df["male"] == 1
            masks["WOMEN"] = merged_df["male"] == 0

        # Age
        if "age" in merged_df.columns:
            masks["Below 50"] = merged_df["age"] < 50
            masks["Above 50"] = merged_df["age"] >= 50

        return masks

    def _prepare_experiment_data(self, exp_name: str, exp_data: dict, demographics: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare merged dataframe for an experiment with simplified probabilities and demographics.

        Returns DataFrame with columns: id_record, pr_omi, pr_nonomi, pr_control,
        age, male, detailed diagnosis columns, true_5group, is_omi_true.
        """
        predictions_df = exp_data.get("predictions_df")
        if predictions_df is None:
            logger.warning(f"No predictions for {exp_name}")
            return None

        config = exp_data["config"]
        simplified = self._get_simplified_probabilities(predictions_df, config)

        # Merge with demographics
        merged = simplified.merge(demographics, on="id_record", how="inner")

        # Add 5-group labels
        merged["true_5group"] = self._get_5group_true_labels(merged)

        # Add binary OMI indicator
        omi_cols_present = [c for c in OMI_COLS if c in merged.columns]
        if omi_cols_present:
            merged["is_omi_true"] = merged[omi_cols_present].max(axis=1)
        else:
            merged["is_omi_true"] = 0

        # Add binary NONOMI indicator
        nonomi_cols_present = [c for c in NONOMI_COLS if c in merged.columns]
        if nonomi_cols_present:
            merged["is_nonomi_true"] = merged[nonomi_cols_present].max(axis=1)
        else:
            merged["is_nonomi_true"] = 0

        return merged

    def create_comparison_table(
        self,
        metrics_to_compare: Optional[List[str]] = None,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Create a comparison table of metrics across experiments."""
        if not self.experiments:
            self.load_all_experiments()

        if not self.experiments:
            logger.warning("No experiments loaded")
            return pd.DataFrame()

        if metrics_to_compare is None:
            metrics_to_compare = [
                "agg_auroc_omi",
                "agg_auroc_nonomi",
                "agg_auroc_control",
                "agg_brier_simplified",
                "accuracy",
                "f1_macro",
                "brier",
            ]

        rows = []
        for exp_name, exp_data in self.experiments.items():
            config = exp_data["config"]
            metrics = exp_data["metrics"]

            row = {
                "experiment": exp_name,
                "exp_id": config.get("exp_id", ""),
                "ensemble_size": config.get("ensemble_size", 1),
                "aggregation": config.get("ensemble_aggregation", "probabilities"),
                "include_age": config.get("include_age", True),
                "include_sex": config.get("include_sex", True),
                "simplified_categories": config.get("use_simplified_categories", False),
            }

            for metric_name in metrics_to_compare:
                value = metrics.get(metric_name)
                row[metric_name] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        if "exp_id" in df.columns:
            df = df.sort_values("exp_id")

        if save_csv:
            csv_path = self.results_dir / "comparison_table.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved comparison table to {csv_path}")

        return df

    def create_cstatistic_bar_chart(self) -> Optional[Any]:
        """
        Create bar chart of C-statistic (AUROC) for OMI by subgroup, colored by experiment.
        Saves to results/cstatistic_by_subgroup.png.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None

        demographics = self._load_demographics()

        subgroup_names = ["OMI", "NONOMI", "NSTEMI-OMI", "MEN", "WOMEN", "Below 50", "Above 50"]
        exp_names = list(self.experiments.keys())

        # Compute AUROC for each experiment × subgroup
        data_rows = []
        for exp_name in exp_names:
            exp_data = self.experiments[exp_name]
            merged = self._prepare_experiment_data(exp_name, exp_data, demographics)
            if merged is None:
                continue

            masks = self._compute_subgroup_masks(merged)

            for sg_name in subgroup_names:
                if sg_name not in masks:
                    continue
                sg_mask = masks[sg_name]
                sg_df = merged[sg_mask]

                if len(sg_df) < 2 or sg_df["is_omi_true"].nunique() < 2:
                    continue

                try:
                    auroc = roc_auc_score(sg_df["is_omi_true"], sg_df["pr_omi"])
                    data_rows.append({
                        "experiment": exp_name,
                        "subgroup": sg_name,
                        "auroc": auroc,
                    })
                except ValueError:
                    continue

        if not data_rows:
            logger.warning("No AUROC data computed for bar chart")
            return None

        plot_df = pd.DataFrame(data_rows)

        fig, ax = plt.subplots(figsize=(14, 6))

        subgroups_present = [sg for sg in subgroup_names if sg in plot_df["subgroup"].values]
        n_subgroups = len(subgroups_present)
        n_experiments = len(exp_names)
        x = np.arange(n_subgroups)
        width = 0.8 / max(n_experiments, 1)

        colors = plt.cm.Set2(np.linspace(0, 1, n_experiments))

        for i, exp_name in enumerate(exp_names):
            exp_df = plot_df[plot_df["experiment"] == exp_name]
            values = []
            for sg in subgroups_present:
                row = exp_df[exp_df["subgroup"] == sg]
                values.append(row["auroc"].values[0] if len(row) > 0 else 0)

            offset = (i - (n_experiments - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=exp_name, color=colors[i])

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(
                        f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_ylabel("C-statistic (AUROC)", fontsize=12)
        ax.set_title("C-statistic (AUROC) for OMI by Subgroup", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(subgroups_present, fontsize=10)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        save_path = self.results_dir / "cstatistic_by_subgroup.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved C-statistic bar chart to {save_path}")
        plt.close(fig)
        return fig

    def create_precision_recall_table(self) -> pd.DataFrame:
        """
        Compute precision/recall for OMI class at thresholds 0.05, 0.01, 0.005.
        Saves to results/precision_recall_thresholds.csv.
        """
        demographics = self._load_demographics()

        rows = []
        for exp_name, exp_data in self.experiments.items():
            merged = self._prepare_experiment_data(exp_name, exp_data, demographics)
            if merged is None:
                continue

            for threshold in THRESHOLDS:
                predicted = self._classify_at_threshold(
                    merged["pr_omi"].values,
                    merged["pr_nonomi"].values,
                    threshold,
                )
                true_omi = merged["is_omi_true"].values

                predicted_omi = (predicted == "OMI").astype(int)

                if predicted_omi.sum() == 0:
                    precision = 0.0
                else:
                    precision = precision_score(true_omi, predicted_omi, zero_division=0)

                recall = recall_score(true_omi, predicted_omi, zero_division=0)

                rows.append({
                    "experiment": exp_name,
                    "threshold": threshold,
                    "precision_omi": round(precision, 4),
                    "recall_omi": round(recall, 4),
                })

        df = pd.DataFrame(rows)
        csv_path = self.results_dir / "precision_recall_thresholds.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved precision/recall table to {csv_path}")

        print("\nPrecision/Recall for OMI at various thresholds:")
        print("=" * 60)
        print(df.to_string(index=False))

        return df

    def create_detailed_confusion_matrices(self) -> Optional[List[str]]:
        """
        Create detailed confusion matrices: 5-row true labels x 3-col predicted labels.
        One image per threshold, each containing all experiments side by side.
        Saves to results/confusion_matrix_threshold_<t>.png.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None

        demographics = self._load_demographics()
        exp_names = list(self.experiments.keys())
        n_exp = len(exp_names)
        saved_paths = []

        for threshold in THRESHOLDS:
            fig, axes = plt.subplots(
                1, n_exp,
                figsize=(6 * n_exp, 5),
                squeeze=False,
            )

            for e_idx, exp_name in enumerate(exp_names):
                ax = axes[0][e_idx]
                exp_data = self.experiments[exp_name]
                merged = self._prepare_experiment_data(exp_name, exp_data, demographics)

                if merged is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.set_title(exp_name)
                    continue

                true_labels = merged["true_5group"].values
                predicted = self._classify_at_threshold(
                    merged["pr_omi"].values,
                    merged["pr_nonomi"].values,
                    threshold,
                )

                # Build 5x3 confusion matrix
                cm = np.zeros((len(FIVE_GROUP_ORDER), len(THREE_GROUP_ORDER)), dtype=int)
                for i, tl in enumerate(FIVE_GROUP_ORDER):
                    for j, pl in enumerate(THREE_GROUP_ORDER):
                        cm[i, j] = int(np.sum((true_labels == tl) & (predicted == pl)))

                # Plot heatmap
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=THREE_GROUP_ORDER,
                    yticklabels=FIVE_GROUP_ORDER,
                    ax=ax, cbar=False,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(exp_name)

            fig.suptitle(f"Confusion Matrices — Threshold {threshold}", fontsize=14)
            plt.tight_layout()
            save_path = self.results_dir / f"confusion_matrix_threshold_{threshold}.png"
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info(f"Saved confusion matrix plot to {save_path}")
            plt.close(fig)
            saved_paths.append(str(save_path))

        return saved_paths

    def create_roc_curves(self) -> Optional[Any]:
        """
        Create ROC curves for OMI and NONOMI outcomes, comparing models.
        Saves to results/roc_curves.png.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None

        demographics = self._load_demographics()

        outcomes = [
            ("OMI", "pr_omi", "is_omi_true"),
            ("NONOMI", "pr_nonomi", "is_nonomi_true"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.experiments)))

        for ax, (outcome_name, pr_col, true_col) in zip(axes, outcomes):
            for i, (exp_name, exp_data) in enumerate(self.experiments.items()):
                merged = self._prepare_experiment_data(exp_name, exp_data, demographics)
                if merged is None:
                    continue

                y_true = merged[true_col].values
                y_score = merged[pr_col].values

                if len(np.unique(y_true)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = roc_auc_score(y_true, y_score)
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f"{exp_name} (AUC={auc_val:.3f})")

            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"ROC Curve — {outcome_name}", fontsize=14)
            ax.legend(loc="lower right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        save_path = self.results_dir / "roc_curves.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curves to {save_path}")
        plt.close(fig)
        return fig

    def generate_all_comparisons(self) -> Dict[str, str]:
        """Generate all comparison outputs (table and plots)."""
        outputs = {}

        self.load_all_experiments()

        if not self.experiments:
            logger.warning("No experiments to compare")
            return outputs

        # Comparison table
        df = self.create_comparison_table()
        outputs["comparison_table"] = str(self.results_dir / "comparison_table.csv")

        # Precision/recall table
        self.create_precision_recall_table()
        outputs["precision_recall"] = str(self.results_dir / "precision_recall_thresholds.csv")

        # Plots
        if PLOTTING_AVAILABLE:
            self.create_cstatistic_bar_chart()
            outputs["cstatistic_bar_chart"] = str(self.results_dir / "cstatistic_by_subgroup.png")

            cm_paths = self.create_detailed_confusion_matrices()
            if cm_paths:
                for path in cm_paths:
                    outputs[f"confusion_matrix_{Path(path).stem}"] = path

            self.create_roc_curves()
            outputs["roc_curves"] = str(self.results_dir / "roc_curves.png")
        else:
            logger.warning("Plotting not available - skipping plot generation")

        # Print summary
        print("\n" + "=" * 60)
        print("Experiment Comparison Summary")
        print("=" * 60)
        print(df.to_string(index=False))
        print("\nOutputs generated:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

        return outputs

    def get_best_experiment(
        self,
        metric: str = "agg_auroc_omi",
        higher_is_better: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Find the best performing experiment by a specific metric."""
        if not self.experiments:
            self.load_all_experiments()

        best_exp = None
        best_value = float("-inf") if higher_is_better else float("inf")
        best_name = None

        for exp_name, exp_data in self.experiments.items():
            value = exp_data["metrics"].get(metric)
            if value is None:
                continue

            if higher_is_better and value > best_value:
                best_value = value
                best_exp = exp_data
                best_name = exp_name
            elif not higher_is_better and value < best_value:
                best_value = value
                best_exp = exp_data
                best_name = exp_name

        if best_name is None:
            logger.warning(f"No experiments found with metric {metric}")
            return None, None

        logger.info(f"Best experiment by {metric}: {best_name} ({best_value:.4f})")
        return best_name, best_exp
