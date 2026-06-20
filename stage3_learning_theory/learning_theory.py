# =============================================================
# STAGE 3 — LEARNING THEORY ANALYSIS
# File: stage3_learning_theory/learning_theory.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model   import (LinearRegression, Ridge,
                                     Lasso, ElasticNet)
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.pipeline        import Pipeline
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.model_selection import (learning_curve,
                                      validation_curve,
                                      cross_val_score,
                                      KFold)
from sklearn.metrics         import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

PALETTE = {
    'train':      '#3498DB',
    'val':        '#E74C3C',
    'gap':        '#F39C12',
    'ideal':      '#2ECC71',
    'complexity': '#9B59B6',
}

# =============================================================
# 3.1  BIAS-VARIANCE DECOMPOSITION (Bootstrap simulation)
# =============================================================

def bias_variance_decomposition(X:         np.ndarray,
                                 y:         np.ndarray,
                                 models:    dict,
                                 n_boot:    int = 80,
                                 test_frac: float = 0.25,
                                 seed:      int = 42) -> pd.DataFrame:
    """
    Empirical bias-variance decomposition.
    For each model, train on n_boot bootstrap samples,
    measure Bias², Variance, and Total Error on fixed test set.
    """

    np.random.seed(seed)
    n          = len(X)
    n_test     = int(n * test_frac)
    test_idx   = np.random.choice(n, n_test, replace=False)
    train_idx  = np.setdiff1d(np.arange(n), test_idx)
    X_test, y_test = X[test_idx], y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    print("=" * 65)
    print("  STAGE 3.1 — Bias-Variance Decomposition")
    print("=" * 65)
    print(f"\n  Bootstrap samples : {n_boot}")
    print(f"  Test set size     : {n_test}")
    print(f"\n  {'Model':<26} {'Bias²':>10} {'Variance':>10} "
          f"{'Noise':>8} {'Total':>10}")
    print(f"  {'─'*26} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

    records = []

    for name, model in models.items():
        preds = np.zeros((n_boot, n_test))

        for b in range(n_boot):
            # Bootstrap sample from training data
            boot_idx = np.random.choice(
                len(X_train), len(X_train), replace=True)
            X_b = X_train[boot_idx]
            y_b = y_train[boot_idx]
            try:
                m = model.__class__(**model.get_params())
                m.fit(X_b, y_b)
                preds[b] = m.predict(X_test)
            except Exception:
                preds[b] = np.mean(y_b)

        # Bias² = (mean prediction - true)²
        mean_pred = preds.mean(axis=0)
        bias2     = np.mean((mean_pred - y_test) ** 2)

        # Variance = mean squared deviation of predictions
        variance  = np.mean(preds.var(axis=0))

        # Irreducible noise estimate
        noise     = np.var(y_test - mean_pred) - variance
        noise     = max(noise, 0.0)

        total = bias2 + variance + noise

        print(f"  {name:<26} {bias2:>10.4f} {variance:>10.4f} "
              f"{noise:>8.4f} {total:>10.4f}")

        records.append({
            'Model':    name,
            'Bias²':    bias2,
            'Variance': variance,
            'Noise':    noise,
            'Total':    total
        })

    df_bv = pd.DataFrame(records).sort_values('Total')

    # ── Stacked Bar Chart ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    x_pos  = np.arange(len(df_bv))
    width  = 0.6
    b2_bar = axes[0].bar(x_pos, df_bv['Bias²'],
                          width, label='Bias²',
                          color='#3498DB', edgecolor='white')
    va_bar = axes[0].bar(x_pos, df_bv['Variance'],
                          width, bottom=df_bv['Bias²'],
                          label='Variance',
                          color='#E74C3C', edgecolor='white')
    no_bar = axes[0].bar(
        x_pos, df_bv['Noise'], width,
        bottom=df_bv['Bias²'] + df_bv['Variance'],
        label='Irreducible Noise',
        color='#95A5A6', edgecolor='white'
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(df_bv['Model'],
                              rotation=35, ha='right', fontsize=9)
    axes[0].set_ylabel("Error Contribution")
    axes[0].set_title("Bias-Variance Decomposition\nStacked by Component",
                       fontweight='bold', fontsize=12)
    axes[0].legend(fontsize=10)

    # ── Bubble Chart: Bias vs Variance ────────────────────────
    scatter = axes[1].scatter(
        df_bv['Bias²'], df_bv['Variance'],
        s=df_bv['Total'] * 3000,
        c=df_bv['Total'],
        cmap='RdYlGn_r',
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5
    )
    for _, row in df_bv.iterrows():
        axes[1].annotate(
            row['Model'], (row['Bias²'], row['Variance']),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, fontweight='bold'
        )
    plt.colorbar(scatter, ax=axes[1], label='Total Error')
    axes[1].set_xlabel("Bias² (Underfitting ↑)")
    axes[1].set_ylabel("Variance (Overfitting ↑)")
    axes[1].set_title("Bias² vs Variance\n(bubble size = total error)",
                       fontweight='bold', fontsize=12)

    # Ideal zone annotation
    axes[1].annotate(
        '🎯 Ideal Zone\n(low bias + low variance)',
        xy=(df_bv['Bias²'].min(), df_bv['Variance'].min()),
        fontsize=9, color='green',
        xytext=(df_bv['Bias²'].mean(), df_bv['Variance'].min()),
        arrowprops=dict(arrowstyle='->', color='green')
    )

    plt.suptitle(
        "Learning Theory — Bias-Variance Analysis on Cloud Execution Time",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage3_bias_variance.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    return df_bv


# =============================================================
# 3.2  LEARNING CURVES — Training Set Size Effect
# =============================================================

def plot_learning_curves(X:      np.ndarray,
                          y:      np.ndarray,
                          models: dict) -> None:
    """Plot learning curves showing bias/variance vs data size."""

    n_models  = len(models)
    n_cols    = 3
    n_rows    = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(18, n_rows * 5))
    axes      = axes.flatten()

    train_sizes = np.linspace(0.1, 1.0, 10)

    for i, (name, model) in enumerate(models.items()):
        ts, tr_sc, va_sc = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        tr_mean = -tr_sc.mean(axis=1)
        tr_std  = tr_sc.std(axis=1)
        va_mean = -va_sc.mean(axis=1)
        va_std  = va_sc.std(axis=1)

        ax = axes[i]
        ax.plot(ts, tr_mean, 'o-',
                color=PALETTE['train'], linewidth=2,
                markersize=6, label='Train RMSE')
        ax.fill_between(ts, tr_mean - tr_std,
                         tr_mean + tr_std,
                         alpha=0.15, color=PALETTE['train'])
        ax.plot(ts, va_mean, 's--',
                color=PALETTE['val'], linewidth=2,
                markersize=6, label='Val RMSE')
        ax.fill_between(ts, va_mean - va_std,
                         va_mean + va_std,
                         alpha=0.15, color=PALETTE['val'])

        # Annotate final gap
        gap = abs(va_mean[-1] - tr_mean[-1])
        ax.annotate(
            f'Gap={gap:.3f}',
            xy=(ts[-1], (va_mean[-1] + tr_mean[-1]) / 2),
            fontsize=8, color=PALETTE['gap'],
            fontweight='bold'
        )

        # Diagnosis label
        if tr_mean[-1] > 0.3 and gap < 0.05:
            diagnosis = '⚠️ Underfitting (High Bias)'
        elif gap > 0.15:
            diagnosis = '⚠️ Overfitting (High Variance)'
        else:
            diagnosis = '✅ Good Fit'

        ax.set_title(f"{name}\n{diagnosis}",
                      fontweight='bold', fontsize=10)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=8)

    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Learning Curves — Cloud Models (Execution Time Prediction)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage3_learning_curves.png",
                dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================
# 3.3  VALIDATION CURVES — Hyperparameter Effect
# =============================================================

def plot_validation_curves(X: np.ndarray,
                            y: np.ndarray) -> None:
    """Show how hyperparameter choice affects bias/variance."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Ridge: alpha (regularization strength) ────────────────
    alphas = np.logspace(-4, 4, 25)
    tr_sc, va_sc = validation_curve(
        Ridge(), X, y,
        param_name='alpha', param_range=alphas,
        cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    _plot_vc_panel(axes[0, 0], alphas,
                   -tr_sc, -va_sc,
                   'Ridge — Regularization Strength (α)',
                   'alpha', log_scale=True,
                   best_label='← High Bias | High Variance →')

    # ── Decision Tree: max_depth ──────────────────────────────
    depths = np.arange(1, 21)
    tr_sc2, va_sc2 = validation_curve(
        DecisionTreeRegressor(random_state=42), X, y,
        param_name='max_depth', param_range=depths,
        cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    _plot_vc_panel(axes[0, 1], depths,
                   -tr_sc2, -va_sc2,
                   'Decision Tree — Max Depth',
                   'max_depth', log_scale=False,
                   best_label='← Underfits | Overfits →')

    # ── SVR: C parameter ──────────────────────────────────────
    C_vals = np.logspace(-2, 3, 20)
    tr_sc3, va_sc3 = validation_curve(
        SVR(kernel='rbf', gamma='scale'), X[:2000], y[:2000],
        param_name='C', param_range=C_vals,
        cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    _plot_vc_panel(axes[1, 0], C_vals,
                   -tr_sc3, -va_sc3,
                   'SVR — Penalty Parameter (C)',
                   'C', log_scale=True,
                   best_label='← Large Margin | Narrow Margin →')

    # ── Random Forest: n_estimators ───────────────────────────
    n_trees = np.array([10, 25, 50, 75, 100,
                         150, 200, 300, 400, 500])
    tr_sc4, va_sc4 = validation_curve(
        RandomForestRegressor(random_state=42,
                               max_features='sqrt'),
        X, y,
        param_name='n_estimators', param_range=n_trees,
        cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    _plot_vc_panel(axes[1, 1], n_trees,
                   -tr_sc4, -va_sc4,
                   'Random Forest — Number of Trees',
                   'n_estimators', log_scale=False,
                   best_label='More trees = less variance')

    plt.suptitle(
        "Validation Curves — Hyperparameter vs Bias/Variance Trade-off",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage3_validation_curves.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def _plot_vc_panel(ax, param_range, tr_sc, va_sc,
                   title, param_name,
                   log_scale=False, best_label=''):
    """Helper: single validation curve panel."""
    tr_mean = tr_sc.mean(axis=1)
    tr_std  = tr_sc.std(axis=1)
    va_mean = va_sc.mean(axis=1)
    va_std  = va_sc.std(axis=1)

    ax.plot(param_range, tr_mean, 'o-',
            color=PALETTE['train'], linewidth=2,
            markersize=5, label='Train RMSE')
    ax.fill_between(param_range,
                     tr_mean - tr_std, tr_mean + tr_std,
                     alpha=0.15, color=PALETTE['train'])
    ax.plot(param_range, va_mean, 's--',
            color=PALETTE['val'], linewidth=2,
            markersize=5, label='Val RMSE')
    ax.fill_between(param_range,
                     va_mean - va_std, va_mean + va_std,
                     alpha=0.15, color=PALETTE['val'])

    # Mark best val score
    best_idx = np.argmin(va_mean)
    ax.axvline(param_range[best_idx],
               color=PALETTE['ideal'], linestyle=':',
               linewidth=2,
               label=f'Best: {param_range[best_idx]:.3g}')

    if log_scale:
        ax.set_xscale('log')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel(f"{param_name}  |  {best_label}",
                   fontsize=8)
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)


# =============================================================
# 3.4  REGULARIZATION PATH STUDY
# =============================================================

def regularization_path_study(X:       np.ndarray,
                                y:       np.ndarray,
                                feat_names: list) -> None:
    """Show how L1/L2 regularization drives coefficients to 0."""

    from sklearn.linear_model import lasso_path, ridge_regression

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── Lasso Path ────────────────────────────────────────────
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=1e-3)
    for i, feat in enumerate(feat_names[:10]):
        axes[0].plot(np.log10(alphas_lasso),
                      coefs_lasso[i],
                      linewidth=1.5, label=feat)
    axes[0].axvline(0, color='black', linestyle='--',
                     linewidth=1, alpha=0.5)
    axes[0].set_title("Lasso (L1) Regularization Path\n"
                       "Coefficients vs log(α)",
                       fontweight='bold', fontsize=11)
    axes[0].set_xlabel("log₁₀(α)  →  More Regularization")
    axes[0].set_ylabel("Coefficient Value")
    axes[0].legend(fontsize=6, loc='upper right',
                    ncol=2)
    axes[0].axhline(0, color='gray', linewidth=0.5)

    # ── Ridge Path ────────────────────────────────────────────
    alphas_ridge = np.logspace(-3, 6, 80)
    coefs_ridge  = []
    for a in alphas_ridge:
        ridge = Ridge(alpha=a).fit(X, y)
        coefs_ridge.append(ridge.coef_)
    coefs_ridge = np.array(coefs_ridge).T

    for i, feat in enumerate(feat_names[:10]):
        axes[1].plot(np.log10(alphas_ridge),
                      coefs_ridge[i],
                      linewidth=1.5, label=feat)
    axes[1].set_title("Ridge (L2) Regularization Path\n"
                       "Coefficients vs log(α)",
                       fontweight='bold', fontsize=11)
    axes[1].set_xlabel("log₁₀(α)  →  More Regularization")
    axes[1].set_ylabel("Coefficient Value")
    axes[1].legend(fontsize=6, loc='upper right', ncol=2)
    axes[1].axhline(0, color='gray', linewidth=0.5)

    # ── Sparsity Comparison ───────────────────────────────────
    alphas_test = np.logspace(-4, 2, 50)
    lasso_sparsity = []
    ridge_sparsity = []

    for a in alphas_test:
        l_coef = Lasso(alpha=a, max_iter=5000).fit(X, y).coef_
        r_coef = Ridge(alpha=a).fit(X, y).coef_
        lasso_sparsity.append((l_coef == 0).mean() * 100)
        ridge_sparsity.append((np.abs(r_coef) < 1e-6).mean() * 100)

    axes[2].plot(np.log10(alphas_test), lasso_sparsity,
                  'b-', linewidth=2.5, label='Lasso (L1) — Sparsity')
    axes[2].plot(np.log10(alphas_test), ridge_sparsity,
                  'r--', linewidth=2.5, label='Ridge (L2) — Near-zero')
    axes[2].set_title("Feature Sparsity vs Regularization\n"
                       "L1 drives features to EXACTLY 0",
                       fontweight='bold', fontsize=11)
    axes[2].set_xlabel("log₁₀(α)")
    axes[2].set_ylabel("% Coefficients = 0")
    axes[2].set_ylim(-5, 105)
    axes[2].legend(fontsize=10)
    axes[2].fill_between(np.log10(alphas_test),
                          lasso_sparsity, ridge_sparsity,
                          alpha=0.1, color='purple',
                          label='Sparsity advantage of L1')

    plt.suptitle(
        "Stage 3 — Regularization Path Analysis on Cloud Features",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage3_regularization_paths.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  ✅ Stage 3 Complete — Learning Theory saved")


# =============================================================
# 3.5  VC DIMENSION DEMONSTRATION
# =============================================================

def vc_dimension_demo(X: np.ndarray, y: np.ndarray) -> None:
    """
    Demonstrate relationship between model complexity,
    VC dimension, and generalization gap.
    """

    models_vc = {
        'Linear (VC≈n+1)':       LogisticRegression(max_iter=500),
        'Poly-2 (VC>n+1)':       Pipeline([
            ('pf', PolynomialFeatures(2)),
            ('lr', LogisticRegression(max_iter=500))]),
        'DTree depth=3 (Low VC)': DecisionTreeClassifier(max_depth=3),
        'DTree depth=15(High VC)':DecisionTreeClassifier(max_depth=15),
        'SVM C=0.1 (Large Margin)':SVC(C=0.1, kernel='rbf',
                                        probability=True),
        'SVM C=1000(Small Margin)': SVC(C=1000, kernel='rbf',
                                         probability=True),
    }

    from sklearn.linear_model    import LogisticRegression
    from sklearn.tree            import DecisionTreeClassifier
    from sklearn.svm             import SVC
    from sklearn.preprocessing   import PolynomialFeatures
    from sklearn.pipeline        import Pipeline

    # Binary classification target
    y_bin = (y > np.median(y)).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes      = axes.flatten()
    train_sizes = np.linspace(0.1, 1.0, 8)

    for ax, (name, model) in zip(axes, models_vc.items()):
        try:
            ts, tr_sc, va_sc = learning_curve(
                model, X[:3000], y_bin[:3000],
                train_sizes=train_sizes, cv=5,
                scoring='accuracy', n_jobs=-1
            )
        except Exception:
            continue

        tr_mean = tr_sc.mean(axis=1)
        va_mean = va_sc.mean(axis=1)
        gap     = tr_mean - va_mean

        ax.plot(ts, tr_mean, 'b-o', linewidth=2,
                label='Train Acc', markersize=5)
        ax.plot(ts, va_mean, 'r--s', linewidth=2,
                label='Val Acc',   markersize=5)
        ax.fill_between(ts, va_mean, tr_mean,
                         alpha=0.15, color='orange',
                         label=f'Gap={gap[-1]:.3f}')
        ax.set_title(f"{name}\nGeneralization Gap={gap[-1]:.3f}",
                      fontweight='bold', fontsize=9)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=8)
        ax.axhline(1.0, color='green', linestyle=':',
                    alpha=0.4, linewidth=1)

    plt.suptitle(
        "VC Dimension Effect — Complexity vs Generalization Gap",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage3_vc_dimension.png",
                dpi=150, bbox_inches='tight')
    plt.show()
