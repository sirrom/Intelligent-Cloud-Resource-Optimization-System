# =============================================================
# STAGE 2C COMPLETION — DBSCAN + PCA + t-SNE + UMAP
# File: stage2c_unsupervised/unsupervised_pipeline.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster       import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture       import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold      import TSNE
from sklearn.metrics       import (silhouette_score,
                                    davies_bouldin_score,
                                    calinski_harabasz_score)
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches    import Ellipse
import matplotlib.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

PALETTE = ['#3498DB','#E74C3C','#2ECC71','#F39C12',
           '#9B59B6','#1ABC9C','#E67E22','#34495E']

# =============================================================
# 2C.2  COMPLETE CLUSTERING SUITE
# =============================================================

def run_all_clustering(X: np.ndarray, k: int = 4) -> dict:
    """Apply K-Means, GMM, Agglomerative, and DBSCAN."""

    results = {}

    # ── K-Means ───────────────────────────────────────────────
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    results['K-Means'] = {
        'labels': km.labels_,
        'model':  km,
        'score':  silhouette_score(X, km.labels_),
        'db':     davies_bouldin_score(X, km.labels_),
        'ch':     calinski_harabasz_score(X, km.labels_)
    }

    # ── GMM via EM Algorithm (Week 9) ─────────────────────────
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           random_state=42, max_iter=300)
    gmm_labels = gmm.fit_predict(X)
    results['GMM (EM)'] = {
        'labels': gmm_labels,
        'model':  gmm,
        'score':  silhouette_score(X, gmm_labels),
        'db':     davies_bouldin_score(X, gmm_labels),
        'ch':     calinski_harabasz_score(X, gmm_labels),
        'bic':    gmm.bic(X),
        'aic':    gmm.aic(X),
        'probs':  gmm.predict_proba(X)
    }

    # ── Agglomerative Clustering (Ward Linkage) ───────────────
    agg        = AgglomerativeClustering(n_clusters=k, linkage='ward')
    agg_labels = agg.fit_predict(X)
    results['Hierarchical (Ward)'] = {
        'labels': agg_labels,
        'model':  agg,
        'score':  silhouette_score(X, agg_labels),
        'db':     davies_bouldin_score(X, agg_labels),
        'ch':     calinski_harabasz_score(X, agg_labels)
    }

    # ── DBSCAN (density-based, no k required) ─────────────────
    dbs        = DBSCAN(eps=0.5, min_samples=5)
    dbs_labels = dbs.fit_predict(X)
    n_clusters = len(set(dbs_labels)) - (1 if -1 in dbs_labels else 0)
    n_noise    = (dbs_labels == -1).sum()
    sil = silhouette_score(X, dbs_labels) \
          if n_clusters > 1 else -1
    results['DBSCAN'] = {
        'labels':     dbs_labels,
        'model':      dbs,
        'score':      sil,
        'n_clusters': n_clusters,
        'n_noise':    n_noise
    }

    # ── Print Summary ─────────────────────────────────────────
    print("=" * 65)
    print("  STAGE 2C — Clustering Results Summary")
    print("=" * 65)
    print(f"\n  {'Algorithm':<24} {'Silhouette':>12} {'DB Index':>10} "
          f"{'CH Score':>12}")
    print(f"  {'─'*24} {'─'*12} {'─'*10} {'─'*12}")
    for name, res in results.items():
        db = res.get('db', float('nan'))
        ch = res.get('ch', float('nan'))
        print(f"  {name:<24} {res['score']:>12.4f} "
              f"{db:>10.4f} {ch:>12.2f}")

    return results


# =============================================================
# 2C.3  DIMENSIONALITY REDUCTION SUITE (Week 10)
# =============================================================

def run_dimensionality_reduction(X:       np.ndarray,
                                  labels:  np.ndarray,
                                  df_orig: pd.DataFrame) -> dict:
    """PCA, Kernel PCA, t-SNE on cloud metrics."""

    print("\n  Running Dimensionality Reduction...")
    embeddings = {}

    # ── PCA ───────────────────────────────────────────────────
    pca_full         = PCA().fit(X)
    cum_var          = np.cumsum(pca_full.explained_variance_ratio_)
    n_95             = np.searchsorted(cum_var, 0.95) + 1
    pca_2d           = PCA(n_components=2).fit_transform(X)
    embeddings['PCA (2D)'] = pca_2d
    print(f"  PCA: {n_95} components explain 95% variance")

    # ── Kernel PCA (RBF) ──────────────────────────────────────
    kpca = KernelPCA(n_components=2, kernel='rbf',
                      gamma=0.1, random_state=42)
    embeddings['Kernel PCA'] = kpca.fit_transform(X)

    # ── t-SNE ─────────────────────────────────────────────────
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, learning_rate='auto',
                init='pca')
    embeddings['t-SNE'] = tsne.fit_transform(X)

    # ── Visualization ─────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                             hspace=0.45, wspace=0.35)

    unique_labels = np.unique(labels)
    colors_map    = {lbl: PALETTE[i % len(PALETTE)]
                     for i, lbl in enumerate(unique_labels)}
    point_colors  = [colors_map[l] for l in labels]

    # ── Row 1: 2D Scatter for Each Reduction Method ──────────
    for col, (method, Z) in enumerate(embeddings.items()):
        ax = fig.add_subplot(gs[0, col])
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=PALETTE[int(lbl) % len(PALETTE)],
                       s=15, alpha=0.5, edgecolors='none',
                       label=f'Cluster {lbl}')
        ax.set_title(f"{method}\nColored by K-Means Cluster",
                      fontweight='bold', fontsize=10)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(fontsize=7, markerscale=2)

    # ── Row 1 Col 4: PCA Scree Plot ───────────────────────────
    ax_scree = fig.add_subplot(gs[0, 3])
    n_show   = min(15, len(pca_full.explained_variance_ratio_))
    ax_scree.bar(range(1, n_show + 1),
                  pca_full.explained_variance_ratio_[:n_show],
                  color='#3498DB', alpha=0.8, edgecolor='white')
    ax2_scree = ax_scree.twinx()
    ax2_scree.plot(range(1, n_show + 1),
                    cum_var[:n_show], 'ro-',
                    linewidth=2, markersize=5,
                    label='Cumulative')
    ax2_scree.axhline(0.95, color='green', linestyle='--',
                       linewidth=1.5, label='95% threshold')
    ax2_scree.set_ylabel("Cumulative Explained Variance")
    ax2_scree.legend(fontsize=8)
    ax_scree.set_title("PCA Scree Plot",
                        fontweight='bold', fontsize=10)
    ax_scree.set_xlabel("Principal Component")
    ax_scree.set_ylabel("Explained Variance Ratio")

    # ── Row 2: GMM Gaussian Ellipses ─────────────────────────
    ax_gmm = fig.add_subplot(gs[1, :2])
    Z      = embeddings['PCA (2D)']

    def draw_ellipse(position, covariance, ax, color, alpha=0.2):
        """Draw 2-sigma ellipse for a 2D Gaussian."""
        if covariance.shape == (2, 2):
            U, s, _ = np.linalg.svd(covariance)
            angle   = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle  = 0
            width  = height = 2 * np.sqrt(covariance)
        ell = Ellipse(xy=position, width=width, height=height,
                       angle=angle, color=color, alpha=alpha,
                       linewidth=2, fill=True)
        ax_gmm.add_patch(ell)

    # Fit GMM on 2D PCA for visualization
    gmm_viz = GaussianMixture(n_components=4,
                               covariance_type='full',
                               random_state=42)
    gmm_viz.fit(Z)
    gmm_viz_labels = gmm_viz.predict(Z)

    for i in range(gmm_viz.n_components):
        mask = gmm_viz_labels == i
        ax_gmm.scatter(Z[mask, 0], Z[mask, 1],
                        c=PALETTE[i], s=15, alpha=0.4,
                        label=f'Component {i}')
        # Project GMM covariance to 2D PCA space
        draw_ellipse(gmm_viz.means_[i],
                      gmm_viz.covariances_[i],
                      ax_gmm, PALETTE[i], alpha=0.15)

    ax_gmm.set_title("GMM with Gaussian Ellipses\n(PCA 2D Projection)",
                      fontweight='bold', fontsize=11)
    ax_gmm.set_xlabel("PC1"); ax_gmm.set_ylabel("PC2")
    ax_gmm.legend(fontsize=9)

    # ── Row 2: BIC/AIC for GMM Component Selection ───────────
    ax_bic = fig.add_subplot(gs[1, 2:])
    k_vals = range(2, 10)
    bics   = [GaussianMixture(n_components=k, random_state=42,
                               covariance_type='full').fit(Z).bic(Z)
               for k in k_vals]
    aics   = [GaussianMixture(n_components=k, random_state=42,
                               covariance_type='full').fit(Z).aic(Z)
               for k in k_vals]
    ax_bic.plot(k_vals, bics, 'bo-', linewidth=2,
                 markersize=7, label='BIC')
    ax_bic.plot(k_vals, aics, 'rs-', linewidth=2,
                 markersize=7, label='AIC')
    ax_bic.axvline(k_vals[np.argmin(bics)], color='blue',
                    linestyle='--', linewidth=1.5,
                    label=f"Best BIC k={k_vals[np.argmin(bics)]}")
    ax_bic.set_title("GMM Model Selection\n(BIC / AIC vs k)",
                      fontweight='bold', fontsize=11)
    ax_bic.set_xlabel("Number of Components")
    ax_bic.set_ylabel("Information Criterion (↓ better)")
    ax_bic.legend(fontsize=9)

    # ── Row 3: Dendrogram (Hierarchical) ──────────────────────
    ax_dend = fig.add_subplot(gs[2, :2])
    sample_idx = np.random.choice(len(X), size=100,
                                   replace=False)
    Z_link = linkage(X[sample_idx], method='ward')
    dendrogram(Z_link, ax=ax_dend,
                color_threshold=0.7 * max(Z_link[:, 2]),
                above_threshold_color='#95A5A6',
                no_labels=True)
    ax_dend.set_title("Hierarchical Clustering Dendrogram\n"
                       "(Ward Linkage — 100 sample points)",
                       fontweight='bold', fontsize=11)
    ax_dend.set_xlabel("Data Points (index)")
    ax_dend.set_ylabel("Merge Distance")

    # ── Row 3: DBSCAN Visualization ───────────────────────────
    ax_dbs = fig.add_subplot(gs[2, 2:])
    dbs        = DBSCAN(eps=0.5, min_samples=5).fit(Z)
    dbs_labels = dbs.labels_
    unique_dbs = set(dbs_labels)

    for lbl in unique_dbs:
        mask   = dbs_labels == lbl
        color  = '#AAAAAA' if lbl == -1 else \
                 PALETTE[lbl % len(PALETTE)]
        marker = 'x' if lbl == -1 else 'o'
        label  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        ax_dbs.scatter(Z[mask, 0], Z[mask, 1],
                        c=color, s=20, alpha=0.5,
                        marker=marker, label=label,
                        edgecolors='none')

    n_cls   = len(unique_dbs) - (1 if -1 in unique_dbs else 0)
    n_noise = (dbs_labels == -1).sum()
    ax_dbs.set_title(f"DBSCAN Clustering (PCA 2D)\n"
                      f"{n_cls} clusters | {n_noise} noise points",
                      fontweight='bold', fontsize=11)
    ax_dbs.set_xlabel("PC1"); ax_dbs.set_ylabel("PC2")
    ax_dbs.legend(fontsize=8, markerscale=1.5)

    plt.suptitle(
        "Stage 2C — Cloud Workload Profiling via Unsupervised Learning",
        fontsize=15, fontweight='bold', y=1.01
    )
    plt.savefig("stage2c_unsupervised_full.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  ✅ Stage 2C Complete — Unsupervised learning saved")
    return embeddings
