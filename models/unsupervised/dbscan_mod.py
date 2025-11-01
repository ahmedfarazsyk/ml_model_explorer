from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run(X, params):
    eps = float(params.get('eps', 0.5))
    min_samples = int(params.get('min_samples', 5))
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = None
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10', s=40)
    ax.set_title('DBSCAN clusters (PCA 2D)')
    plt.tight_layout()
    return labels, sil, fig