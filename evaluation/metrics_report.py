# ==============================================================================
# evaluation/metrics_report.py
# Capstone Project — Unified Metrics Reporter
# Generates per-stage metric tables, classification reports,
# regression diagnostics, clustering scores, and RL summaries.
# ==============================================================================

from __future__ import annotations

import os
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy  as np
import pandas as pd
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    # Classification
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    # Clustering
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "blue":   "#3498DB",
    "red":    "#E74C3C",
    "green":  "#2ECC71",
    "orange": "#F39C12",
    "purple": "#9B59B6",
    "teal":   "#1ABC9C",
    "grey":   "#95A5A6",
    "dark":   "#2C3E50",
}

_OUTPUT_DIR = "reports"
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. REGRESSION METRICS
# ==============================================================================

class RegressionMetricsReport:
    """
    Computes and formats all standard regression metrics for one or more models.

    Usage
    -----
    >>> rpt = RegressionMetricsReport()
    >>> rpt.add_model("Random Forest", y_test, y_pred_rf)
    >>> rpt.add_model("Ridge",         y_test, y_pred_ridge)
    >>> df = rpt.summary()
    >>> rpt.plot(save_path="reports/regression_metrics.png")
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    # ── Add a model ───────────────────────────────────────────────────────────
    def add_model(
        self,
        name:    str,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_train_pred: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register predictions for one model.

        Parameters
        ----------
        name          : model identifier string
        y_true        : ground-truth test labels
        y_pred        : test set predictions
        y_train       : training labels        (optional, for overfit gap)
        y_train_pred  : training predictions   (optional, for overfit gap)
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse  = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        evs  = explained_variance_score(y_true, y_pred)

        # Mean Absolute Percentage Error (MAPE)
        mask = y_true != 0
        mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                      / y_true[mask]))) * 100
                if mask.any() else float("nan"))

        # Overfit gap  (train R² − test R²)
        overfit_gap = float("nan")
        if y_train is not None and y_train_pred is not None:
            r2_train    = r2_score(y_train, y_train_pred)
            overfit_gap = round(float(r2_train - r2), 4)

        self._records.append({
            "Model":         name,
            "R²":            round(r2,   4),
            "Adj R²":        round(self._adj_r2(r2, len(y_true), 1), 4),
            "RMSE":          round(rmse, 4),
            "MAE":           round(mae,  4),
            "MAPE (%)":      round(mape, 2),
            "Expl Var":      round(evs,  4),
            "Overfit Gap":   overfit_gap,
            "_y_true":       y_true,
            "_y_pred":       y_pred,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    def summary(self, sort_by: str = "R²",
                ascending: bool = False) -> pd.DataFrame:
        """
        Returns a clean, sorted summary DataFrame.
        Drops internal columns (_y_true, _y_pred).
        """
        df = (pd.DataFrame(self._records)
                .drop(columns=["_y_true", "_y_pred"], errors="ignore")
                .sort_values(sort_by, ascending=ascending)
                .reset_index(drop=True))
        print("\n" + "═" * 72)
        print("  REGRESSION METRICS SUMMARY")
        print("═" * 72)
        print(df.to_string(index=False))
        df.to_csv(f"{_OUTPUT_DIR}/regression_metrics.csv", index=False)
        print(f"\n  Saved → {_OUTPUT_DIR}/regression_metrics.csv")
        return df

    # ── Visualisation ─────────────────────────────────────────────────────────
    def plot(self, save_path: str = "reports/regression_metrics.png") -> None:
        """
        6-panel figure:
          [0] Bar chart — R² per model
          [1] Bar chart — RMSE per model
          [2] Scatter   — Actual vs Predicted (best model)
          [3] Histogram — Residuals (best model)
          [4] Bar chart — MAE per model
          [5] Bar chart — MAPE per model
        """
        if not self._records:
            print("  [RegressionMetricsReport] No models added yet.")
            return

        df_sorted = (pd.DataFrame(self._records)
                       .sort_values("R²", ascending=False)
                       .reset_index(drop=True))
        best = df_sorted.iloc[0]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Regression Metrics Report",
                      fontsize=16, fontweight="bold", y=1.01)

        ax = axes[0, 0]
        colors = [C["blue"] if i == 0 else C["grey"]
                  for i in range(len(df_sorted))]
        bars = ax.barh(df_sorted["Model"], df_sorted["R²"],
                        color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("R² Score")
        ax.set_title("R² Score — All Models", fontweight="bold")
        for bar, val in zip(bars, df_sorted["R²"]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax = axes[0, 1]
        bars = ax.barh(df_sorted["Model"], df_sorted["RMSE"],
                        color=[C["red"] if i == 0 else C["grey"]
                                for i in range(len(df_sorted))],
                        edgecolor="white")
        ax.set_xlabel("RMSE")
        ax.set_title("RMSE — All Models (lower ↓)", fontweight="bold")
        for bar, val in zip(bars, df_sorted["RMSE"]):
            ax.text(val + df_sorted["RMSE"].max() * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax    = axes[0, 2]
        yt    = best["_y_true"]
        yp    = best["_y_pred"]
        ax.scatter(yt, yp, alpha=0.35, s=14,
                    color=C["blue"], edgecolors="none")
        lims = [min(yt.min(), yp.min()),
                max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1.8, label="Perfect fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted\n({best['Model']})",
                      fontweight="bold")
        ax.legend(fontsize=8)

        ax  = axes[1, 0]
        res = yt - yp
        ax.hist(res, bins=40, color=C["purple"], edgecolor="white",
                 alpha=0.80)
        ax.axvline(0, color="red", linewidth=1.8, linestyle="--")
        ax.axvline(res.mean(), color="orange", linewidth=1.8,
                    linestyle="--", label=f"Mean={res.mean():.3f}")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residual Distribution\n({best['Model']})",
                      fontweight="bold")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        bars = ax.barh(df_sorted["Model"], df_sorted["MAE"],
                        color=[C["orange"] if i == 0 else C["grey"]
                                for i in range(len(df_sorted))],
                        edgecolor="white")
        ax.set_xlabel("MAE")
        ax.set_title("MAE — All Models (lower ↓)", fontweight="bold")
        for bar, val in zip(bars, df_sorted["MAE"]):
            ax.text(val + df_sorted["MAE"].max() * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax = axes[1, 2]
        valid_mape = df_sorted.dropna(subset=["MAPE (%)"])
        bars = ax.barh(valid_mape["Model"], valid_mape["MAPE (%)"],
                        color=[C["teal"] if i == 0 else C["grey"]
                                for i in range(len(valid_mape))],
                        edgecolor="white")
        ax.set_xlabel("MAPE (%)")
        ax.set_title("MAPE % — All Models (lower ↓)", fontweight="bold")
        for bar, val in zip(bars, valid_mape["MAPE (%)"]):
            ax.text(val + valid_mape["MAPE (%)"].max() * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  ✅ Regression metrics plot saved → {save_path}")

    # ── Static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _adj_r2(r2: float, n: int, p: int) -> float:
        if n <= p + 1:
            return float("nan")
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# ==============================================================================
# 2. CLASSIFICATION METRICS
# ==============================================================================

class ClassificationMetricsReport:
    """
    Computes and formats all standard classification metrics.

    Usage
    -----
    >>> rpt = ClassificationMetricsReport(class_names=["Low","Med","High"])
    >>> rpt.add_model("Random Forest", y_test, y_pred_rf)
    >>> df = rpt.summary()
    >>> rpt.plot(save_path="reports/clf_metrics.png")
    """

    def __init__(self,
                 class_names: Optional[List[str]] = None):
        self._records:     List[Dict[str, Any]] = []
        self.class_names = class_names

    # ── Add a model ───────────────────────────────────────────────────────────
    def add_model(
        self,
        name:       str,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        y_prob:     Optional[np.ndarray] = None,
        cv_scores:  Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        name      : model identifier
        y_true    : ground-truth test labels
        y_pred    : predicted labels
        y_prob    : predicted probabilities (for AUC-ROC)
        cv_scores : cross-validation accuracy scores (optional)
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        avg     = "weighted"
        acc     = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1      = f1_score(y_true, y_pred, average=avg, zero_division=0)
        prec    = precision_score(y_true, y_pred, average=avg,
                                   zero_division=0)
        rec     = recall_score(y_true, y_pred, average=avg,
                                zero_division=0)
        mcc     = matthews_corrcoef(y_true, y_pred)
        kappa   = cohen_kappa_score(y_true, y_pred)

        # AUC-ROC (multi-class OvR if needed)
        auc = float("nan")
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob,
                                     multi_class="ovr",
                                     average="weighted")
            except Exception:
                pass

        cv_mean = float(np.mean(cv_scores)) if cv_scores is not None else float("nan")
        cv_std  = float(np.std(cv_scores))  if cv_scores is not None else float("nan")

        self._records.append({
            "Model":        name,
            "Accuracy":     round(acc,     4),
            "Bal Accuracy": round(bal_acc, 4),
            "F1 (weighted)":round(f1,      4),
            "Precision":    round(prec,    4),
            "Recall":       round(rec,     4),
            "MCC":          round(mcc,     4),
            "Cohen Kappa":  round(kappa,   4),
            "AUC-ROC":      round(auc, 4) if not np.isnan(auc) else "—",
            "CV Mean":      round(cv_mean, 4) if not np.isnan(cv_mean) else "—",
            "CV Std":       round(cv_std,  4) if not np.isnan(cv_std)  else "—",
            "_y_true":      y_true,
            "_y_pred":      y_pred,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    def summary(self, sort_by: str = "Accuracy",
                ascending: bool = False) -> pd.DataFrame:
        df = (pd.DataFrame(self._records)
                .drop(columns=["_y_true", "_y_pred"], errors="ignore")
                .sort_values(sort_by, ascending=ascending)
                .reset_index(drop=True))
        print("\n" + "═" * 72)
        print("  CLASSIFICATION METRICS SUMMARY")
        print("═" * 72)
        print(df.to_string(index=False))
        df.to_csv(f"{_OUTPUT_DIR}/classification_metrics.csv", index=False)
        print(f"\n  Saved → {_OUTPUT_DIR}/classification_metrics.csv")
        return df

    # ── Visualisation ─────────────────────────────────────────────────────────
    def plot(self, best_model_name: Optional[str] = None,
             save_path: str = "reports/clf_metrics.png") -> None:
        """
        6-panel figure:
          [0] Accuracy bar chart
          [1] F1 / Precision / Recall grouped bar
          [2] Confusion matrix (best model)
          [3] MCC & Cohen-Kappa bar chart
          [4] CV Mean ± Std
          [5] Per-class F1 heatmap (best model)
        """
        if not self._records:
            print("  [ClassificationMetricsReport] No models added.")
            return

        df_s = (pd.DataFrame(self._records)
                  .drop(columns=["_y_true", "_y_pred"], errors="ignore")
                  .sort_values("Accuracy", ascending=False)
                  .reset_index(drop=True))

        # Pick best model record for detailed panels
        best_name = best_model_name or df_s.iloc[0]["Model"]
        best_rec  = next((r for r in self._records
                           if r["Model"] == best_name),
                          self._records[0])

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle("Classification Metrics Report",
                      fontsize=16, fontweight="bold", y=1.01)

        # Panel 1 — Accuracy
        ax = axes[0, 0]
        ax.barh(df_s["Model"], df_s["Accuracy"],
                 color=[C["blue"] if i == 0 else C["grey"]
                         for i in range(len(df_s))],
                 edgecolor="white")
        ax.axvline(1, color="green", linestyle=":", linewidth=1.4)
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy — All Models", fontweight="bold")
        for i, val in enumerate(df_s["Accuracy"]):
            ax.text(float(val) + 0.002, i, f"{val:.4f}",
                    va="center", fontsize=8)

        # Panel 2 — F1 / Precision / Recall grouped
        ax  = axes[0, 1]
        x   = np.arange(len(df_s))
        w   = 0.26
        ax.bar(x - w, df_s["F1 (weighted)"].astype(float),
                width=w, label="F1",        color=C["blue"])
        ax.bar(x,     df_s["Precision"].astype(float),
                width=w, label="Precision", color=C["green"])
        ax.bar(x + w, df_s["Recall"].astype(float),
                width=w, label="Recall",    color=C["orange"])
        ax.set_xticks(x)
        ax.set_xticklabels(df_s["Model"], rotation=25,
                            ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title("F1 / Precision / Recall",   fontweight="bold")
        ax.legend(fontsize=8)

        # Panel 3 — Confusion matrix (best model)
        ax = axes[0, 2]
        cm = confusion_matrix(best_rec["_y_true"],
                               best_rec["_y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     ax=ax,
                     xticklabels=self.class_names or "auto",
                     yticklabels=self.class_names or "auto",
                     linewidths=0.4, linecolor="white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix\n({best_name})",
                      fontweight="bold")

        # Panel 4 — MCC & Kappa
        ax = axes[1, 0]
        x2 = np.arange(len(df_s))
        ax.bar(x2 - 0.18, df_s["MCC"].astype(float),
                width=0.35, label="MCC",         color=C["purple"])
        ax.bar(x2 + 0.18, df_s["Cohen Kappa"].astype(float),
                width=0.35, label="Cohen Kappa", color=C["teal"])
        ax.set_xticks(x2)
        ax.set_xticklabels(df_s["Model"], rotation=25,
                            ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Score")
        ax.set_title("MCC & Cohen Kappa",         fontweight="bold")
        ax.legend(fontsize=8)

        # Panel 5 — CV Mean ± Std
        ax = axes[1, 1]
        cv_rows = df_s[df_s["CV Mean"] != "—"].copy()
        if not cv_rows.empty:
            cv_rows["CV Mean"] = cv_rows["CV Mean"].astype(float)
            cv_rows["CV Std"]  = cv_rows["CV Std"].astype(float)
            ax.bar(cv_rows["Model"],
                    cv_rows["CV Mean"],
                    yerr=cv_rows["CV Std"],
                    color=C["red"], alpha=0.75,
                    edgecolor="white",
                    capsize=5, error_kw=dict(linewidth=1.5))
            ax.set_ylabel("CV Accuracy")
            ax.set_xticklabels(cv_rows["Model"],
                                rotation=25, ha="right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No CV data supplied",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
        ax.set_title("Cross-Validation Accuracy (Mean ± Std)",
                      fontweight="bold")

        # Panel 6 — Per-class F1 heatmap
        ax  = axes[1, 2]
        rpt = classification_report(
            best_rec["_y_true"],
            best_rec["_y_pred"],
            output_dict=True,
            zero_division=0,
            target_names=self.class_names,
        )
        cls_keys = [k for k in rpt.keys()
                     if k not in ("accuracy",
                                   "macro avg",
                                   "weighted avg")]
        cls_data = pd.DataFrame(
            {k: rpt[k] for k in cls_keys},
            index=["precision", "recall", "f1-score"],
        ).T
        sns.heatmap(cls_data[["precision", "recall", "f1-score"]],
                     annot=True, fmt=".3f", cmap="YlGn",
                     ax=ax, linewidths=0.4,
                     vmin=0, vmax=1)
        ax.set_title(f"Per-Class Metrics\n({best_name})",
                      fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  ✅ Classification metrics plot saved → {save_path}")


# ==============================================================================
# 3. CLUSTERING METRICS
# ==============================================================================

class ClusteringMetricsReport:
    """
    Computes and formats clustering evaluation metrics.

    Usage
    -----
    >>> rpt = ClusteringMetricsReport()
    >>> rpt.add_result("KMeans",      X, labels_km)
    >>> rpt.add_result("Agglomerative", X, labels_ag)
    >>> df = rpt.summary()
    >>> rpt.plot()
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add_result(
        self,
        name:       str,
        X:          np.ndarray,
        labels:     np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> None:
        X      = np.asarray(X)
        labels = np.asarray(labels)
        unique = len(set(labels) - {-1})   # exclude DBSCAN noise

        if unique < 2:
            print(f"  [ClusteringMetrics] {name}: only {unique} "
                  f"cluster(s) — skipping metric computation.")
            return

        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)),
                                random_state=42)
        dbi = davies_bouldin_score(X, labels)
        chs = calinski_harabasz_score(X, labels)

        noise_pct = (np.sum(labels == -1) / len(labels) * 100
                     if -1 in labels else 0.0)

        self._records.append({
            "Algorithm":       name,
            "N Clusters":      n_clusters or unique,
            "Silhouette":      round(sil, 4),
            "Davies-Bouldin":  round(dbi, 4),
            "Calinski-Harabasz": round(chs, 1),
            "Noise %":         round(noise_pct, 2),
            "_labels":         labels,
        })

    def summary(self, sort_by: str = "Silhouette",
                ascending: bool = False) -> pd.DataFrame:
        df = (pd.DataFrame(self._records)
                .drop(columns=["_labels"], errors="ignore")
                .sort_values(sort_by, ascending=ascending)
                .reset_index(drop=True))
        print("\n" + "═" * 72)
        print("  CLUSTERING METRICS SUMMARY")
        print("═" * 72)
        print(df.to_string(index=False))
        df.to_csv(f"{_OUTPUT_DIR}/clustering_metrics.csv", index=False)
        print(f"\n  Saved → {_OUTPUT_DIR}/clustering_metrics.csv")
        return df

    def plot(self,
             save_path: str = "reports/clustering_metrics.png") -> None:
        """3-panel: Silhouette · Davies-Bouldin · Calinski-Harabasz."""
        if not self._records:
            print("  [ClusteringMetricsReport] No results added.")
            return

        df_s = (pd.DataFrame(self._records)
                  .drop(columns=["_labels"], errors="ignore")
                  .sort_values("Silhouette", ascending=False)
                  .reset_index(drop=True))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Clustering Metrics Report",
                      fontsize=15, fontweight="bold")

        metrics = [
            ("Silhouette",         C["blue"],   "higher ↑", False),
            ("Davies-Bouldin",     C["red"],    "lower ↓",  True),
            ("Calinski-Harabasz",  C["green"],  "higher ↑", False),
        ]
        for ax, (metric, color, note, lower_better) in zip(axes, metrics):
            sorted_df = df_s.sort_values(metric, ascending=lower_better)
            bars = ax.barh(sorted_df["Algorithm"],
                            sorted_df[metric],
                            color=color, alpha=0.80,
                            edgecolor="white")
            ax.set_title(f"{metric}\n({note})", fontweight="bold")
            ax.set_xlabel(metric)
            for bar, val in zip(bars, sorted_df[metric]):
                ax.text(val + sorted_df[metric].max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  ✅ Clustering metrics plot saved → {save_path}")


# ==============================================================================
# 4. RL METRICS
# ==============================================================================

class RLMetricsReport:
    """
    Formats and visualises RL evaluation results.

    Usage
    -----
    >>> rpt = RLMetricsReport(action_names=["Scale DOWN","HOLD",
    ...                                     "Scale UP","MIGRATE"])
    >>> rpt.add_agent("Q-Learning", ql_log, ql_eval)
    >>> rpt.add_agent("DQN",        dqn_log, dqn_eval)
    >>> rpt.add_agent("Random",     None,    rand_eval)
    >>> df = rpt.summary()
    >>> rpt.plot()
    """

    def __init__(self, action_names: Optional[List[str]] = None):
        self._records:    List[Dict[str, Any]] = []
        self.action_names = action_names or [
            "Scale DOWN", "HOLD", "Scale UP", "MIGRATE"]

    def add_agent(
        self,
        name:     str,
        train_log: Optional[Dict],
        eval_res:  Dict,
    ) -> None:
        n_ep = len(train_log["ep_rewards"]) if train_log else 0
        best = max(train_log["ep_rewards"]) if train_log else float("nan")
        avg_last = (float(np.mean(train_log["ep_rewards"][-100:]))
                    if train_log else float("nan"))

        self._records.append({
            "Agent":            name,
            "Train Episodes":   n_ep,
            "Best Ep Reward":   round(best, 3) if not np.isnan(best) else "—",
            "Final Avg(100)":   round(avg_last, 3) if not np.isnan(avg_last) else "—",
            "Eval Mean Reward": round(eval_res["mean_reward"], 3),
            "Eval Std Reward":  round(eval_res["std_reward"],  3),
            "Eval Max Reward":  round(eval_res["max_reward"],  3),
            "Eval Min Reward":  round(eval_res["min_reward"],  3),
            "SLA Violation %":  round(eval_res["sla_rate"] * 100, 2),
            "_train_log":       train_log,
            "_eval_res":        eval_res,
        })

    def summary(self) -> pd.DataFrame:
        df = (pd.DataFrame(self._records)
                .drop(columns=["_train_log", "_eval_res"], errors="ignore")
                .reset_index(drop=True))
        print("\n" + "═" * 72)
        print("  RL METRICS SUMMARY")
        print("═" * 72)
        print(df.to_string(index=False))
        df.to_csv(f"{_OUTPUT_DIR}/rl_metrics.csv", index=False)
        print(f"\n  Saved → {_OUTPUT_DIR}/rl_metrics.csv")
        return df

    def plot(self,
             save_path: str = "reports/rl_metrics.png") -> None:
        """4-panel: eval reward box · SLA bar · action pie charts."""
        if not self._records:
            print("  [RLMetricsReport] No agents added.")
            return

        n_agents = len(self._records)
        fig = plt.figure(figsize=(22, 14))
        gs  = gridspec.GridSpec(2, n_agents + 1, figure=fig,
                                 hspace=0.45, wspace=0.38)

        colors = [C["blue"], C["red"], C["grey"],
                   C["orange"], C["purple"]]

        # Row 0 — Eval reward boxplot (spans all columns)
        ax_box = fig.add_subplot(gs[0, :])
        reward_data  = [r["_eval_res"]["rewards"] for r in self._records]
        agent_labels = [r["Agent"]               for r in self._records]
        bp = ax_box.boxplot(
            reward_data, labels=agent_labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.6),
            capprops=dict(linewidth=1.6),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        for i, (rewards, jx) in enumerate(
                zip(reward_data, range(1, n_agents + 1))):
            jitter = np.random.normal(jx, 0.06, len(rewards))
            ax_box.scatter(jitter, rewards, s=16,
                            alpha=0.35, color=colors[i], zorder=3)
        ax_box.set_ylabel("Episode Reward")
        ax_box.set_title("Evaluation Reward Distribution "
                          "(50 greedy episodes per agent)",
                          fontweight="bold", fontsize=12)

        # Row 1 — Action distribution pie per agent
        for i, rec in enumerate(self._records):
            ax = fig.add_subplot(gs[1, i])
            dist = rec["_eval_res"].get("action_dist",
                                         np.ones(4) / 4)
            ax.pie(dist,
                    labels=self.action_names,
                    autopct="%1.1f%%",
                    colors=[C["blue"], C["green"],
                             C["red"], C["orange"]],
                    startangle=90,
                    wedgeprops=dict(edgecolor="white",
                                     linewidth=0.8))
            ax.set_title(f"{rec['Agent']}\nAction Distribution",
                          fontweight="bold", fontsize=10)

        # Row 1, last panel — SLA violation bar
        ax_sla = fig.add_subplot(gs[1, n_agents])
        sla_vals  = [r["SLA Violation %"] for r in self._records]
        sla_names = [r["Agent"]           for r in self._records]
        bars = ax_sla.bar(sla_names, sla_vals,
                           color=colors[:n_agents],
                           edgecolor="white", width=0.5)
        ax_sla.axhline(10, color="red", linestyle="--",
                        linewidth=1.8, label="10% Target")
        ax_sla.set_ylabel("SLA Violation %")
        ax_sla.set_title("SLA Violation\n(lower ↓)",
                          fontweight="bold")
        ax_sla.legend(fontsize=8)
        for bar, val in zip(bars, sla_vals):
            ax_sla.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center",
                fontsize=9, fontweight="bold",
                color="red" if val > 10 else "green",
            )

        fig.suptitle("RL Metrics Report",
                      fontsize=15, fontweight="bold", y=1.01)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  ✅ RL metrics plot saved → {save_path}")


# ==============================================================================
# 5. MASTER REPORT GENERATOR
# ==============================================================================

def generate_master_report(
    reg_report:  Optional[RegressionMetricsReport]     = None,
    clf_report:  Optional[ClassificationMetricsReport] = None,
    clust_report:Optional[ClusteringMetricsReport]     = None,
    rl_report:   Optional[RLMetricsReport]             = None,
    project_name:str = "Cloud Resource Optimisation — ML Capstone",
    output_path: str = "reports/master_metrics_report.json",
) -> Dict:
    """
    Consolidates all stage metrics into a single JSON artefact.

    Returns
    -------
    dict   Full nested report dictionary.
    """
    report: Dict[str, Any] = {
        "project":   project_name,
        "stages":    {},
    }

    if reg_report and reg_report._records:
        best = max(reg_report._records,
                   key=lambda x: x.get("R²", -999))
        report["stages"]["regression"] = {
            "best_model": best["Model"],
            "best_r2":    best["R²"],
            "best_rmse":  best["RMSE"],
        }

    if clf_report and clf_report._records:
        best = max(clf_report._records,
                   key=lambda x: x.get("Accuracy", 0))
        report["stages"]["classification"] = {
            "best_model":  best["Model"],
            "best_acc":    best["Accuracy"],
            "best_f1":     best["F1 (weighted)"],
        }

    if clust_report and clust_report._records:
        best = max(clust_report._records,
                   key=lambda x: x.get("Silhouette", -1))
        report["stages"]["clustering"] = {
            "best_algorithm": best["Algorithm"],
            "silhouette":     best["Silhouette"],
            "davies_bouldin": best["Davies-Bouldin"],
        }

    if rl_report and rl_report._records:
        best = max(rl_report._records,
                   key=lambda x: (
                       x["Eval Mean Reward"]
                       if isinstance(x["Eval Mean Reward"], float)
                       else -999
                   ))
        report["stages"]["rl"] = {
            "best_agent":      best["Agent"],
            "eval_mean_reward":best["Eval Mean Reward"],
            "sla_violation_%": best["SLA Violation %"],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "═" * 72)
    print("  MASTER METRICS REPORT")
    print("═" * 72)
    print(json.dumps(report, indent=2))
    print(f"\n  ✅ Master report saved → {output_path}")
    return report
