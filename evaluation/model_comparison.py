# ==============================================================================  
# evaluation/model_comparison.py  
# Capstone Project — Cross-Stage Model Comparison Engine  
# Side-by-side ranking, radar charts, improvement matrices,  
# and the final cross-stage leaderboard.  
# ==============================================================================  

from __future__ import annotations  

import os  
import warnings  
from typing import Any, Dict, List, Optional, Tuple  

import numpy  as np  
import pandas as pd  
import matplotlib.pyplot   as plt  
import matplotlib.gridspec as gridspec  
import matplotlib.patches  as mpatches  
from matplotlib.patches import FancyArrowPatch  
import seaborn as sns  
from matplotlib.projections.polar import PolarAxes  
import matplotlib.transforms as mtransforms  

warnings.filterwarnings("ignore")  

C = {  
    "blue":   "#3498DB",  
    "red":    "#E74C3C",  
    "green":  "#2ECC71",  
    "orange": "#F39C12",  
    "purple": "#9B59B6",  
    "teal":   "#1ABC9C",  
    "grey":   "#95A5A6",  
    "dark":   "#2C3E50",  
    "yellow": "#F1C40F",  
}  

_OUTPUT_DIR = "reports"  
os.makedirs(_OUTPUT_DIR, exist_ok=True)  


# ==============================================================================  
# 1. REGRESSION COMPARATOR  
# ==============================================================================  

class RegressionComparator:  
    """  
    Side-by-side comparison of multiple regression models with  
    ranking, radar chart, and residual comparison.  

    Usage  
    -----  
    >>> rc = RegressionComparator()  
    >>> rc.add("Random Forest", y_test, preds_rf, train_r2=0.98)  
    >>> rc.add("Ridge",         y_test, preds_ridge)  
    >>> rc.rank()  
    >>> rc.plot()  
    """  

    def __init__(self):  
        self._data: List[Dict] = []  

    def add(  
        self,  
        name:     str,  
        y_true:   np.ndarray,  
        y_pred:   np.ndarray,  
        train_r2: float = float("nan"),  
    ) -> None:  
        from sklearn.metrics import (mean_squared_error,  
                                      mean_absolute_error,  
                                      r2_score)  
        y_true = np.asarray(y_true).ravel()  
        y_pred = np.asarray(y_pred).ravel()  
        r2   = float(r2_score(y_true, y_pred))  
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  
        mae  = float(mean_absolute_error(y_true, y_pred))  
        res  = y_true - y_pred  

        self._data.append({  
            "Model":         name,  
            "Test R²":       round(r2,   4),  
            "Train R²":      round(train_r2, 4),  
            "Overfit Gap":   round(float(train_r2 - r2), 4)  
                              if not np.isnan(train_r2) else float("nan"),  
            "RMSE":          round(rmse, 4),  
            "MAE":           round(mae,  4),  
            "_res":          res,  
            "_y_true":       y_true,  
            "_y_pred":       y_pred,  
        })  

    def rank(self,  
             weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:  
        """  
        Composite score = Σ w_i · normalised_metric_i  
        Default weights: R²=0.4, RMSE=0.3 (inverted), MAE=0.3 (inverted).  
        """  
        w = weights or {"Test R²": 0.40,  
                         "RMSE":    0.30,  
                         "MAE":     0.30}  
        df = pd.DataFrame(self._data).drop(  
            columns=["_res", "_y_true", "_y_pred"], errors="ignore")  

        scores = np.zeros(len(df))  
        for metric, weight in w.items():  
            col  = df[metric].astype(float)  
            mn   = col.min()  
            mx   = col.max()  
            norm = (col - mn) / (mx - mn + 1e-9)  
            # For error metrics: invert normalisation  
            if metric in ("RMSE", "MAE", "Overfit Gap"):  
                norm = 1.0 - norm  
            scores += weight * norm  

        df["Composite Score"] = np.round(scores, 4)  
        df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)  
        df.insert(0, "Rank", range(1, len(df) + 1))  

        print("\n" + "═" * 72)  
        print("  REGRESSION MODEL RANKING")  
        print("═" * 72)  
        print(df.to_string(index=False))  
        df.to_csv(f"{_OUTPUT_DIR}/regression_ranking.csv", index=False)  
        return df  

    def plot(self,  
             save_path: str = "reports/regression_comparison.png") -> None:  
        """4-panel: metric bars · scatter overlay · residuals · radar."""  
        if len(self._data) < 2:  
            print("  [RegressionComparator] Need ≥2 models.")  
            return  

        df = (pd.DataFrame(self._data)  
                .drop(columns=["_res","_y_true","_y_pred"], errors="ignore")  
                .sort_values("Test R²", ascending=False)  
                .reset_index(drop=True))  

        fig, axes = plt.subplots(2, 2, figsize=(18, 13))  
        fig.suptitle("Regression Model Comparison",  
                      fontsize=16, fontweight="bold", y=1.01)  
        colors = list(C.values())  

        # — Metric bar chart (R², RMSE, MAE grouped) —  
        ax  = axes[0, 0]  
        x   = np.arange(len(df))  
        w   = 0.25  
        ax.bar(x - w, df["Test R²"].astype(float), width=w,  
                label="Test R²",  color=C["blue"])  
        ax.bar(x,     df["RMSE"].astype(float),    width=w,  
                label="RMSE",     color=C["red"],  alpha=0.80)  
        ax.bar(x + w, df["MAE"].astype(float),     width=w,  
                label="MAE",      color=C["orange"],alpha=0.80)  
        ax.set_xticks(x)  
        ax.set_xticklabels(df["Model"], rotation=20, ha="right",  
                            fontsize=9)  
        ax.legend(fontsize=9)  
        ax.set_title("R² · RMSE · MAE per Model", fontweight="bold")  

        # — Actual vs Predicted scatter for all models —  
        ax = axes[0, 1]  
        for i, rec in enumerate(self._data):  
            ax.scatter(rec["_y_true"], rec["_y_pred"],  
                        alpha=0.25, s=12,  
                        color=colors[i % len(colors)],  
                        label=rec["Model"])  
        all_vals = np.concatenate(  
            [np.concatenate([r["_y_true"], r["_y_pred"]])  
              for r in self._data])  
        lims = [all_vals.min(), all_vals.max()]  
        ax.plot(lims, lims, "k--", linewidth=1.8,  
                 label="Perfect fit")  
        ax.set_xlabel("Actual")  
        ax.set_ylabel("Predicted")  
        ax.set_title("Actual vs Predicted — All Models",  
                      fontweight="bold")  
        ax.legend(fontsize=7, ncol=2)  

        # — Residual boxplot —  
        ax = axes[1, 0]  
        bp = ax.boxplot(  
            [r["_res"] for r in self._data],  
            labels=[r["Model"] for r in self._data],  
            patch_artist=True,  
            medianprops=dict(color="black", linewidth=2),  
            whiskerprops=dict(linewidth=1.5),  
        )  
        for patch, col in zip(bp["boxes"], colors):  
            patch.set_facecolor(col)  
            patch.set_alpha(0.65)  
        ax.axhline(0, color="red", linestyle="--", linewidth=1.6)  
        ax.set_xticklabels([r["Model"] for r in self._data],  
                            rotation=20, ha="right", fontsize=9)  
        ax.set_ylabel("Residual")  
        ax.set_title("Residual Distribution — All Models",  
                      fontweight="bold")  

        # — Radar chart —  
        ax = axes[1, 1]  
        _radar_chart(ax, df,  
                      metrics=["Test R²", "RMSE", "MAE"],  
                      invert=["RMSE", "MAE"],  
                      title="Normalised Radar Chart")  

        plt.tight_layout()  
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  
        plt.show()  
        print(f"  ✅ Regression comparison saved → {save_path}")  


# ==============================================================================  
# 2. CLASSIFICATION COMPARATOR  
# ==============================================================================  

class ClassificationComparator:  
    """  
    Side-by-side comparison of multiple classification models.  

    Usage  
    -----  
    >>> cc = ClassificationComparator()  
    >>> cc.add("Random Forest", y_test, preds_rf, cv_scores_rf)  
    >>> cc.add("SVM",           y_test, preds_svm)  
    >>> cc.rank()  
    >>> cc.plot()  
    """  

    def __init__(self):  
        self._data: List[Dict] = []  

    def add(  
        self,  
        name:      str,  
        y_true:    np.ndarray,  
        y_pred:    np.ndarray,  
        cv_scores: Optional[np.ndarray] = None,  
    ) -> None:  
        from sklearn.metrics import (accuracy_score, f1_score,  
                                      precision_score, recall_score,  
                                      matthews_corrcoef)  
        y_true = np.asarray(y_true).ravel()  
        y_pred = np.asarray(y_pred).ravel()  
        self._data.append({  
            "Model":       name,  
            "Accuracy":    round(accuracy_score(y_true, y_pred), 4),  
            "F1":          round(f1_score(y_true, y_pred,  
                                           average="weighted",  
                                           zero_division=0), 4),  
            "Precision":   round(precision_score(y_true, y_pred,  
                                                  average="weighted",  
                                                  zero_division=0), 4),  
            "Recall":      round(recall_score(y_true, y_pred,  
                                               average="weighted",  
                                               zero_division=0), 4),  
            "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),  
            "CV Mean":     round(float(np.mean(cv_scores)), 4)  
                            if cv_scores is not None else float("nan"),  
            "CV Std":      round(float(np.std(cv_scores)),  4)  
                            if cv_scores is not None else float("nan"),  
            "_y_true":     y_true,  
            "_y_pred":     y_pred,  
        })  

    def rank(self) -> pd.DataFrame:  
        df = (pd.DataFrame(self._data)  
                .drop(columns=["_y_true","_y_pred"], errors="ignore")  
                .sort_values("F1", ascending=False)  
                .reset_index(drop=True))  
        df.insert(0, "Rank", range(1, len(df) + 1))  
        print("\n" + "═" * 72)  
        print("  CLASSIFICATION MODEL RANKING")  
        print("═" * 72)  
        print(df.to_string(index=False))  
        df.to_csv(f"{_OUTPUT_DIR}/classification_ranking.csv", index=False)  
        return df  

    def plot(self,  
             save_path: str = "reports/classification_comparison.png") -> None:  
        """4-panel: metric bars · CV comparison · confusion matrices · radar."""  
        if len(self._data) < 2:  
            print("  [ClassificationComparator] Need ≥2 models.")  
            return  

        df = (pd.DataFrame(self._data)  
                .drop(columns=["_y_true","_y_pred"], errors="ignore")  
                .sort_values("Accuracy", ascending=False)  
                .reset_index(drop=True))  

        n_models = len(df)  
        fig = plt.figure(figsize=(22, 14))  
        gs  = gridspec.GridSpec(2, max(n_models, 2),  
                                 figure=fig,  
                                 hspace=0.48, wspace=0.36)  
        colors = list(C.values())  

        # — Row 0: metric bar groups (spans half) —  
        ax = fig.add_subplot(gs[0, :n_models // 2 + 1])  
        x  = np.arange(n_models)  
        w  = 0.18  
        for j, (metric, col) in enumerate(  
                zip(["Accuracy","F1","Precision","Recall"],  
                    [C["blue"],C["green"],C["orange"],C["red"]])):  
            ax.bar(x + j * w, df[metric].astype(float),  
                    width=w, label=metric, color=col, alpha=0.82)  
        ax.set_xticks(x + 1.5 * w)  
        ax.set_xticklabels(df["Model"], rotation=20,  
                            ha="right", fontsize=9)  
        ax.set_ylim(0, 1.15)  
        ax.legend(fontsize=8, ncol=2)  
        ax.set_title("Accuracy · F1 · Precision · Recall",  
                      fontweight="bold")  

        # — CV Mean ± Std (spans remaining) —  
        ax = fig.add_subplot(gs[0, n_models // 2 + 1:])  
        cv_rows = df.dropna(subset=["CV Mean"]).copy()  
        if not cv_rows.empty:  
            ax.bar(cv_rows["Model"],  
                    cv_rows["CV Mean"].astype(float),  
                    yerr=cv_rows["CV Std"].astype(float),  
                    color=C["purple"], alpha=0.75,  
                    edgecolor="white", capsize=6,  
                    error_kw=dict(linewidth=1.6))  
            ax.set_ylabel("CV Accuracy")  
            ax.set_xticklabels(cv_rows["Model"],  
                                rotation=20, ha="right",  
                                fontsize=9)  
        else:  
            ax.text(0.5, 0.5, "No CV data supplied",  
                    ha="center", va="center",  
                    transform=ax.transAxes, fontsize=11)  
        ax.set_title("Cross-Validation (Mean ± Std)",  
                      fontweight="bold")  

        # — Row 1: confusion matrix per model —  
        from sklearn.metrics import confusion_matrix as cm_fn  
        for i, rec in enumerate(self._data):  
            if i >= gs.get_geometry()[1]:  
                break  
            ax = fig.add_subplot(gs[1, i])  
            cm = cm_fn(rec["_y_true"], rec["_y_pred"])  
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  
                         ax=ax, linewidths=0.4,  
                         linecolor="white", cbar=False)  
            ax.set_xlabel("Predicted", fontsize=8)  
            ax.set_ylabel("Actual",    fontsize=8)  
            ax.set_title(rec["Model"], fontweight="bold", fontsize=9)  

        fig.suptitle("Classification Model Comparison",  
                      fontsize=15, fontweight="bold", y=1.01)  
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  
        plt.show()  
        print(f"  ✅ Classification comparison saved → {save_path}")  


# ==============================================================================  
# 3. CROSS-STAGE LEADERBOARD  
# ==============================================================================  

class CrossStageLeaderboard:  
    """  
    Aggregates the best result from every stage into a unified  
    scorecard with a visual dashboard.  

    Usage  
    -----  
    >>> lb = CrossStageLeaderboard()  
    >>> lb.add_stage("Regression",     "Random Forest",  
    ...              {"R²": 0.92, "RMSE": 0.04})  
    >>> lb.add_stage("Classification", "GBM",  
    ...              {"Accuracy": 0.94, "F1": 0.93})  
    >>> lb.add_stage("Clustering",     "KMeans-6",  
    ...              {"Silhouette": 0.61})  
    >>> lb.add_stage("Learning Theory","Ridge (tuned)",  
    ...              {"Bias²": 0.01, "Variance": 0.005})  
    >>> lb.add_stage("RL (DQN)",       "DQN",  
    ...              {"Eval Reward": 48.3, "SLA Viol %": 4.1})  
    >>> lb.plot()  
    """  

    def __init__(self):  
        self._stages: List[Dict] = []  

    def add_stage(  
        self,  
        stage_name: str,  
        best_model: str,  
        metrics:    Dict[str, float],  
        status:     str = "✅ Complete",  
    ) -> None:  
        row = {  
            "Stage":      stage_name,  
            "Best Model": best_model,  
            "Status":     status,  
        }  
        row.update(metrics)  
        self._stages.append(row)  

    def table(self) -> pd.DataFrame:  
        df = pd.DataFrame(self._stages)  
        print("\n" + "═" * 80)  
        print("  CROSS-STAGE LEADERBOARD")  
        print("═" * 80)  
        print(df.to_string(index=False))  
        df.to_csv(f"{_OUTPUT_DIR}/cross_stage_leaderboard.csv", index=False)  
        print(f"\n  Saved → {_OUTPUT_DIR}/cross_stage_leaderboard.csv")  
        return df  

    def plot(self,  
             save_path: str = "reports/cross_stage_leaderboard.png") -> None:  
        """  
        Master 6-panel summary figure integrating all stages.  
        """  
        if not self._stages:  
            print("  [CrossStageLeaderboard] No stages added.")  
            return  

        fig = plt.figure(figsize=(26, 20))  
        gs  = gridspec.GridSpec(3, 3, figure=fig,  
                                 hspace=0.55, wspace=0.40)  

        # ── Panel 1: Title / Stage summary table ─────────────────────────────  
        ax = fig.add_subplot(gs[0, :])  
        ax.axis("off")  

        df = pd.DataFrame(self._stages)  
        col_names = df.columns.tolist()  
        cell_text = df.values.tolist()  

        tbl = ax.table(  
            cellText  = cell_text,  
            colLabels = col_names,  
            cellLoc   = "center",  
            loc       = "center",  
        )  
        tbl.auto_set_font_size(False)  
        tbl.set_fontsize(9.5)  
        tbl.scale(1, 2.0)  

        # Style header  
        for j in range(len(col_names)):  
            tbl[0, j].set_facecolor(C["dark"])  
            tbl[0, j].set_text_props(color="white",  
                                      fontweight="bold")  
        # Alternate row colouring  
        for i in range(1, len(df) + 1):  
            bg = "#EAF3FB" if i % 2 == 0 else "white"  
            for j in range(len(col_names)):  
                tbl[i, j].set_facecolor(bg)  

        ax.set_title("Cross-Stage Leaderboard — Best Results per Stage",  
                      fontweight="bold", fontsize=14, pad=12)  

        # ── Panel 2: Stage status badges ─────────────────────────────────────  
        ax = fig.add_subplot(gs[1, 0])  
        ax.axis("off")  
        for i, row in enumerate(self._stages):  
            color = C["green"] if "✅" in row["Status"] else C["orange"]  
            ax.add_patch(mpatches.FancyBboxPatch(  
                (0.05, 0.82 - i * 0.17), 0.90, 0.13,  
                boxstyle="round,pad=0.02",  
                facecolor=color, alpha=0.20,  
                edgecolor=color, linewidth=2,  
                transform=ax.transAxes,  
            ))  
            ax.text(0.10, 0.885 - i * 0.17,  
                     f"{row['Status']}  {row['Stage']}",  
                     transform=ax.transAxes,  
                     fontsize=11, fontweight="bold",  
                     va="center", color=C["dark"])  
            ax.text(0.10, 0.855 - i * 0.17,  
                     f"  → Best: {row['Best Model']}",  
                     transform=ax.transAxes,  
                     fontsize=9, va="center",  
                     color=C["grey"])  
        ax.set_title("Stage Completion Status",  
                      fontweight="bold", fontsize=11)  

        # ── Panel 3: Key
