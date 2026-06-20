# =============================================================  
# STAGE 2C — UNSUPERVISED LEARNING PIPELINE  
# File: stage2c_unsupervised/unsupervised_pipeline.py  
# =============================================================  

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.cluster        import (KMeans, DBSCAN,  
                                     AgglomerativeClustering)  
from sklearn.mixture        import GaussianMixture  
from sklearn.decomposition  import PCA, KernelPCA  
from sklearn.manifold       import TSNE  
from sklearn.metrics        import (silhouette_score,  
                                     davies_bouldin_score,  
                                     calinski_harabasz_score)  
from scipy.cluster.hierarchy import (dendrogram, linkage)  
from scipy.spatial.distance  import cdist  
import warnings  
warnings.filterwarnings('ignore')  

# =============================================================  
# 2C.1  OPTIMAL K SEARCH (Elbow + Silhouette)  
# =============================================================  

def find_optimal_k(X: np.ndarray,  
                   k_range: range = range(2, 12)) -> int:  
    """Determine optimal number of clusters."""  

    inertias    = []  
    silhouettes = []  
    db_scores   = []  
    ch_scores   = []  

    for k in k_range:  
        km     = KMeans(n_clusters=k, random_state=42,  
                         n_init=10)  
        labels = km.fit_predict(X)  
        inertias.append(km.inertia_)  
        silhouettes.append(silhouette_score(X, labels))  
        db_scores.append(davies_bouldin_score(X, labels))  
        ch_scores.append(calinski_harabasz_score(X, labels))  

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  

    axes[0,0].plot(k_range, inertias, 'bo-', linewidth=2,  
                    markersize=8)  
    axes[0,0].set_title("Elbow Method — Inertia",  
                          fontweight='bold', fontsize=11)  
    axes[0,0].set_xlabel("Number of Clusters k")  
    axes[0,0].set_ylabel("Inertia (Within-Cluster SSE)")  

    # Annotate elbow point  
    diffs  = np.diff(inertias)  
    diffs2 = np.diff(diffs)  
    elbow_k = list(k_range)[np.argmax(diffs2) + 2]  
    axes[0,0].axvline(elbow_k, color='red', linestyle='--',  
                       linewidth=2, label=f'Elbow at k={elbow_k}')  
    axes[0,0].legend(fontsize=10)  

    axes[0,1].plot(k_range, silhouettes, 'go-', linewidth=2,  
                    markersize=8)  
    best_k_sil = list(k_range)[np.argmax(silhouettes)]  
    axes[0,1].axvline(best_k_sil, color='red', linestyle='--',  
                       linewidth=2,  
                       label=f'Best k={best_k_sil}')  
    axes[0,1].set_title("Silhouette Score",  
                          fontweight='bold', fontsize=11)  
    axes[0,1].set_xlabel("Number of Clusters k")  
    axes[0,1].set_ylabel("Silhouette Score (↑ better)")  
    axes[0,1].legend(fontsize=10)  

    axes[1,0].plot(k_range, db_scores, 'ro-', linewidth=2,  
                    markersize=8)  
    axes[1,0].set_title("Davies-Bouldin Index (↓ better)",  
                          fontweight='bold', fontsize=11)  
    axes[1,0].set_xlabel("k"); axes[1,0].set_ylabel("DB Index")  

    axes[1,1].plot(k_range, ch_scores, 'mo-', linewidth=2,  
                    markersize=8)  
    axes[1,1].set_title("Calinski-Harabász Score (↑ better)",  
                          fontweight='bold', fontsize=11)  
    axes[1,1].set_xlabel("k"); axes[1,1].set_ylabel("CH Score")  

    plt.suptitle(  
        "Optimal Cluster Selection — Cloud Workload Profiles",  
        fontsize=14, fontweight='bold'  
    )  
    plt.tight_layout()  
    plt.savefig("stage2c_optimal_k.png", dpi=150, bbox_inches='tight')  
    plt.show()  

    return best_k_sil  


# =============================================================  
# 2C.2  FULL CLUSTERING SUITE  
# =============================================================  

def run_all_clustering(X: np.ndarray, k: int = 4) -> dict:  
    """Apply all clustering algorithms."""  

    results = {}  

    # ── K-Means ───────────────────────────────────────────────  
    km = KMeans(n_clusters=k, random_state=42, n_init=10)  
    results['K-Means'] = {  
        'labels': km.fit_predict(X),  
        'model':  km,  
        'score':  silhouette_score(X, km.labels_)  
    }  

    # ── GMM (EM Algorithm) ────────────────────────────────────  
    gmm = GaussianMixture(n_components=k, covariance_type='full',  
                           random_state=42, max_iter=200)  
    gmm_labels = gmm.fit_predict(X)  
    results['GMM (EM)'] = {  
        'labels':   gmm_labels,  
        'model':    gmm,  
        'score':    silhouette_score(X, gmm_labels),  
        'bic':      gmm.bic(X),  
        'aic':      gmm.aic(X),  
        'probs':    gmm.predict_proba(X)  
    }  

    # ── Agglomerative (Ward) ──────────────────────────────────  
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')  
    agg_labels = agg.fit_predict(X)  
    results['Hierarchical (Ward)'] = {  
        'labels': agg_labels,  
        'model':  agg,  
        'score':  silhouette_score(X,
