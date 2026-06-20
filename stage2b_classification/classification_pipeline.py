# =============================================================
# STAGE 2B — CLASSIFICATION PIPELINE
# File: stage2b_classification/classification_pipeline.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model   import LogisticRegression
from sklearn.naive_bayes    import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     AdaBoostClassifier)
from sklearn.svm            import SVC
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.metrics        import (accuracy_score, f1_score,
                                     classification_report,
                                     confusion_matrix,
                                     roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 2B.1  MODEL ZOO — All Classification Models
# =============================================================

def build_classification_models():
    return {
        # ── Discriminative Models (Week 3, 4) ────────────────
        'Logistic Regression':   LogisticRegression(
                                     max_iter=500, random_state=42),
        'LDA (Generative)':      LinearDiscriminantAnalysis(),
        'QDA (Generative)':      QuadraticDiscriminantAnalysis(),
        'Naïve Bayes':           GaussianNB(),

        # ── Non-Parametric (Week 5) ───────────────────────────
        'KNN (k=5)':             KNeighborsClassifier(n_neighbors=5),
        'KNN (k=15)':            KNeighborsClassifier(n_neighbors=15),

        # ── Tree-Based (Week 6) ───────────────────────────────
        'Decision Tree':         DecisionTreeClassifier(
                                     max_depth=8, random_state=42),
        'Random Forest':         RandomForestClassifier(
                                     n_estimators=100, random_state=42),
        'Gradient Boosting':     GradientBoostingClassifier(
                                     n_estimators=100, random_state=42),
        'AdaBoost':              AdaBoostClassifier(
                                     n_estimators=100, random_state=42),

        # ── SVM (Week 8) ──────────────────────────────────────
        'SVM (RBF)':             SVC(kernel='rbf', C=10,
                                     probability=True, random_state=42),
        'SVM (Linear)':          SVC(kernel='linear', C=1,
                                     probability=True, random_state=42),
    }


# =============================================================
# 2B.2  TRAIN & EVALUATE ALL CLASSIFIERS
# =============================================================

def train_and_evaluate_classification(data: dict) -> pd.DataFrame:

    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    X_te, y_te = data['X_test'],  data['y_test']

    models  = build_classification_models()
    records = []

    print("=" * 80)
    print("  STAGE 2B — Classification Model Comparison")
    print("=" * 80)
    print(f"\n  {'Model':<24} {'Train Acc':>10} {'Val Acc':>10} "
          f"{'Test Acc':>10} {'F1':>9} {'CV Mean':>10}")
    print(f"  {'─'*24} {'─'*10} {'─'*10} {'─'*10} {'─'*9} {'─'*10}")

    for name, model in models.items():
        model.fit(X_tr, y_tr)

        train_acc = model.score(X_tr, y_tr)
        val_acc   = model.score(X_va, y_va)
        test_acc  = model.score(X_te, y_te)
        f1        = f1_score(y_te, model.predict(X_te),
                             average='weighted')
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5,
                                     scoring='accuracy')

        print(f"  {name:<24} {train_acc:>10.4f} {val_acc:>10.4f} "
              f"{test_acc:>10.4f} {f1:>9.4f} "
              f"{cv_scores.mean():>10.4f}±{cv_scores.std():.3f}")

        records.append({
            'Model':     name,
            'Train Acc': train_acc,
            'Val Acc':   val_acc,
            'Test Acc':  test_acc,
            'F1':        f1,
            'CV Mean':   cv_scores.mean(),
            'CV Std':    cv_scores.std(),
            'Overfit':   train_acc - test_acc,
            'fitted':    model
        })

    return pd.DataFrame(records).sort_values('Test Acc', ascending=False)


# =============================================================
# 2B.3  GENERATIVE vs DISCRIMINATIVE DEEP DIVE (Week 4)
# =============================================================

def generative_vs_discriminative_analysis(data: dict) -> None:
    """Detailed comparison with learning curves."""

    gen_models  = {
        'Naïve Bayes':  GaussianNB(),
        'LDA':          LinearDiscriminantAnalysis(),
        'QDA':          QuadraticDiscriminantAnalysis(),
    }
    disc_models = {
        'Logistic Reg': LogisticRegression(max_iter=500, random_state=42),
        'SVM (RBF)':    SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=50, random_state=42),
    }

    X_all = np.vstack([data['X_train'], data['X_val']])
    y_all = np.concatenate([data['y_train'], data['y_val']])

    train_sizes = np.linspace(0.1, 1.0, 8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    all_models = {**{'[G] ' + k: v for k, v in gen_models.items()},
                  **{'[D] ' + k: v for k, v in disc_models.items()}}

    for ax, (name, model) in zip(axes.flatten(), all_models.items()):
        train_sz, train_sc, val_sc = learning_curve(
            model, X_all, y_all,
            train_sizes=train_sizes,
            cv=5, scoring='accuracy',
            n_jobs=-1
        )
        train_mean = train_sc.mean(axis=1)
        train_std  = train_sc.std(axis=1)
        val_mean   = val_sc.mean(axis=1)
        val_std    = val_sc.std(axis=1)

        tag   = '🔴 Generative' if name.startswith('[G]') \
                else '🔵 Discriminative'
        color = '#E74C3C' if name.startswith('[G]') else '#3498DB'

        ax.plot(train_sz, train_mean, 'o-', color=color,
                linewidth=2, label='Train', markersize=5)
        ax.fill_between(train_sz,
                         train_mean - train_std,
                         train_mean + train_std,
                         alpha=0.15, color=color)
        ax.plot(train_sz, val_mean, 's--', color=color,
                linewidth=2, label='Val', alpha=0.7, markersize=5)
        ax.fill_between(train_sz,
                         val_mean - val_std,
                         val_mean + val_std,
                         alpha=0.1, color=color)

        final_acc = val_mean[-1]
        ax.set_title(f"{name}\n{tag}  |  Val Acc={final_acc:.3f}",
                      fontweight='bold', fontsize=10)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Generative vs Discriminative — Learning Curves on Cloud Data",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage2b_gen_vs_disc.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================
# 2B.4  CONFUSION MATRIX DASHBOARD
# =============================================================

def plot_confusion_matrices(results_df: pd.DataFrame,
                             data:        dict) -> None:
    """Plot confusion matrices for top 6 models."""

    top6   = results_df.head(6)
    labels = np.unique(data['y_test'])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for ax, (_, row) in zip(axes.flatten(), top6.iterrows()):
        model = row['fitted']
        y_pred = model.predict(data['X_test'])
        cm     = confusion_matrix(data['y_test'], y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    ax=ax, linewidths=0.5,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': '% of True Class'})
        ax.set_title(f"{row['Model']}\nTest Acc={row['Test Acc']:.4f}  "
                     f"F1={row['F1']:.4f}",
                     fontweight='bold', fontsize=10)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.suptitle(
        "Top 6 Models — Confusion Matrices (Task Status Classification)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage2b_confusion_matrices.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  ✅ Stage 2B Complete")
