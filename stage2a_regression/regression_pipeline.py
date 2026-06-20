# =============================================================
# STAGE 2A — REGRESSION PIPELINE
# File: stage2a_regression/regression_pipeline.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model  import (LinearRegression, Ridge, Lasso,
                                    ElasticNet, BayesianRidge)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline      import Pipeline
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import (RandomForestRegressor,
                                    GradientBoostingRegressor)
from sklearn.svm           import SVR
from sklearn.metrics       import (mean_squared_error, r2_score,
                                    mean_absolute_error)
from sklearn.model_selection import cross_val_score, learning_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 2A.1  MODEL ZOO — All Regression Models
# =============================================================

def build_regression_models():
    """Return dictionary of all regression models to compare."""
    return {
        # ── Parametric Models (Week 2) ────────────────────────
        'Linear Regression':     LinearRegression(),
        'Ridge (L2)':            Ridge(alpha=1.0),
        'Lasso (L1)':            Lasso(alpha=0.01),
        'ElasticNet':            ElasticNet(alpha=0.01, l1_ratio=0.5),
        'Bayesian Ridge':        BayesianRidge(),
        'Polynomial (deg=3)':    Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('lr',   LinearRegression())
        ]),

        # ── Non-Parametric Models (Week 5) ────────────────────
        'KNN (k=5)':             KNeighborsRegressor(n_neighbors=5),
        'KNN (k=15)':            KNeighborsRegressor(n_neighbors=15),
        'Decision Tree':         DecisionTreeRegressor(
                                     max_depth=8, random_state=42),

        # ── Ensemble Models (Week 6) ──────────────────────────
        'Random Forest':         RandomForestRegressor(
                                     n_estimators=100, random_state=42),
        'Gradient Boosting':     GradientBoostingRegressor(
                                     n_estimators=100, random_state=42),

        # ── SVM (Week 8) ──────────────────────────────────────
        'SVR (RBF)':             SVR(kernel='rbf', C=10, gamma='scale'),
        'SVR (Linear)':          SVR(kernel='linear', C=1.0),
    }


# =============================================================
# 2A.2  TRAIN & EVALUATE ALL MODELS
# =============================================================

def train_and_evaluate_regression(data: dict) -> pd.DataFrame:
    """Train all regression models and collect metrics."""

    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    X_te, y_te = data['X_test'],  data['y_test']

    models  = build_regression_models()
    records = []

    print("=" * 75)
    print("  STAGE 2A — Regression Model Comparison")
    print("=" * 75)
    print(f"\n  {'Model':<24} {'Train R²':>9} {'Val R²':>9} "
          f"{'Test R²':>9} {'RMSE':>9} {'MAE':>9}")
    print(f"  {'─'*24} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

    for name, model in models.items():
        model.fit(X_tr, y_tr)

        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)
        y_te_pred = model.predict(X_te)

        train_r2 = r2_score(y_tr, y_tr_pred)
        val_r2   = r2_score(y_va, y_va_pred)
        test_r2  = r2_score(y_te, y_te_pred)
        rmse     = np.sqrt(mean_squared_error(y_te, y_te_pred))
        mae      = mean_absolute_error(y_te, y_te_pred)

        print(f"  {name:<24} {train_r2:>9.4f} {val_r2:>9.4f} "
              f"{test_r2:>9.4f} {rmse:>9.4f} {mae:>9.4f}")

        records.append({
            'Model':    name,
            'Train R²': train_r2,
            'Val R²':   val_r2,
            'Test R²':  test_r2,
            'RMSE':     rmse,
            'MAE':      mae,
            'Overfit':  train_r2 - test_r2
        })

    return pd.DataFrame(records).sort_values('Test R²', ascending=False)


# =============================================================
# 2A.3  NEURAL NETWORK REGRESSOR (Week 7)
# =============================================================

def build_nn_regressor(input_dim: int) -> keras.Model:
    """Deep neural network for regression on cloud metrics."""

    inputs = keras.Input(shape=(input_dim,), name='cloud_features')

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, activation='relu')(x)

    output = layers.Dense(1, activation='linear', name='execution_time')(x)

    model = keras.Model(inputs, output, name='CloudRegressor')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


def train_nn_regressor(data: dict) -> dict:
    """Train neural network and return history + metrics."""

    model = build_nn_regressor(data['X_train'].shape[1])
    model.summary()

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=100,
        batch_size=64,
        callbacks=callback_list,
        verbose=0
    )

    y_pred = model.predict(data['X_test'], verbose=0).flatten()
    test_r2   = r2_score(data['y_test'], y_pred)
    test_rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

    print(f"\n  🧠 Neural Network Results:")
    print(f"     Test R²   : {test_r2:.4f}")
    print(f"     Test RMSE : {test_rmse:.4f}")

    return {'model': model, 'history': history,
            'test_r2': test_r2, 'test_rmse': test_rmse,
            'y_pred': y_pred}


# =============================================================
# 2A.4  VISUALIZATION SUITE
# =============================================================

def visualize_regression_results(results_df:  pd.DataFrame,
                                  nn_results:  dict,
                                  data:        dict) -> None:
    """Comprehensive regression results dashboard."""

    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                             hspace=0.4, wspace=0.35)

    COLORS = ['#3498DB','#E74C3C','#2ECC71','#F39C12',
              '#9B59B6','#1ABC9C','#E67E22','#34495E']

    # ── Plot 1: Model R² Comparison ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    top_n  = results_df.head(10)
    colors = [COLORS[i % len(COLORS)] for i in range(len(top_n))]
    bars   = ax1.barh(top_n['Model'], top_n['Test R²'],
                       color=colors, edgecolor='white',
                       linewidth=0.5)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Test R² Score", fontsize=11)
    ax1.set_title("Regression Models — Test R² Comparison",
                   fontweight='bold', fontsize=12)
    for bar, val in zip(bars, top_n['Test R²']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9,
                 fontweight='bold')
    ax1.axvline(0.8, color='green', linestyle='--',
                linewidth=1, alpha=0.7, label='R²=0.80 target')
    ax1.legend(fontsize=9)

    # ── Plot 2: Bias-Variance (Overfit) Chart ─────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(results_df['Train R²'],
                results_df['Test R²'],
                c=[COLORS[i % len(COLORS)]
                   for i in range(len(results_df))],
                s=80, edgecolors='white', linewidths=1, zorder=3)
    # Perfect line
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1,
             alpha=0.5, label='Perfect (no overfit)')
    ax2.fill_between([0, 1], [0, 1], [0, 0.85],
                     alpha=0.05, color='red',
                     label='Overfit zone')
    for _, row in results_df.iterrows():
        ax2.annotate(row['Model'][:12],
                     (row['Train R²'], row['Test R²']),
                     textcoords="offset points",
                     xytext=(4, 4), fontsize=6)
    ax2.set_xlabel("Train R²"); ax2.set_ylabel("Test R²")
    ax2.set_title("Bias-Variance Check\n(Train vs Test R²)",
                   fontweight='bold', fontsize=11)
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, 1.1); ax2.set_ylim(0, 1.1)

    # ── Plot 3: Neural Network Training Curves ────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    hist = nn_results['history'].history
    ax3.plot(hist['loss'],     label='Train Loss',
             color='#3498DB', linewidth=2)
    ax3.plot(hist['val_loss'], label='Val Loss',
             color='#E74C3C', linewidth=2, linestyle='--')
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Huber Loss")
    ax3.set_title("Neural Network\nTraining Curves",
                   fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9)

    # ── Plot 4: NN Predicted vs Actual ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    y_true = np.array(data['y_test'])
    y_pred = nn_results['y_pred']
    ax4.scatter(y_true, y_pred, alpha=0.3, s=10,
                color='#9B59B6', edgecolors='none')
    lim = [min(y_true.min(), y_pred.min()),
           max(y_true.max(), y_pred.max())]
    ax4.plot(lim, lim, 'r--', linewidth=2, label='Perfect prediction')
    ax4.set_xlabel("Actual Execution Time")
    ax4.set_ylabel("Predicted Execution Time")
    ax4.set_title(f"NN: Actual vs Predicted\nR²={nn_results['test_r2']:.4f}",
                   fontweight='bold', fontsize=11)
    ax4.legend(fontsize=9)

    # ── Plot 5: Residual Distribution ────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    residuals = y_true - y_pred
    ax5.hist(residuals, bins=40, color='#1ABC9C',
              alpha=0.7, edgecolor='white', density=True)
    from scipy import stats
    x_r = np.linspace(residuals.min(), residuals.max(), 200)
    ax5.plot(x_r, stats.norm.pdf(x_r, residuals.mean(),
                                   residuals.std()),
              'r-', linewidth=2, label='Normal fit')
    ax5.axvline(0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel("Residual (Actual − Predicted)")
    ax5.set_ylabel("Density")
    ax5.set_title("NN Residual Distribution",
                   fontweight='bold', fontsize=11)
    ax5.legend(fontsize=9)

    # ── Plot 6: RMSE Comparison Bar Chart ─────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(len(results_df))
    width = 0.35
    bars1 = ax6.bar(x_pos - width/2, results_df['RMSE'],
                     width, label='RMSE',
                     color='#3498DB', alpha=0.8, edgecolor='white')
    bars2 = ax6.bar(x_pos + width/2, results_df['MAE'],
                     width, label='MAE',
                     color='#E74C3C', alpha=0.8, edgecolor='white')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(results_df['Model'],
                         rotation=35, ha='right', fontsize=9)
    ax6.set_ylabel("Error")
    ax6.set_title("RMSE & MAE Comparison Across All Models",
                   fontweight='bold', fontsize=12)
    ax6.legend(fontsize=10)

    plt.suptitle("Stage 2A — Cloud Execution Time Regression Results",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig("stage2a_regression_results.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  ✅ Stage 2A Complete — Regression results saved")
