from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run(X, params):
    n_clusters = int(params.get('n_clusters', 3))
    linkage = params.get('linkage', 'ward')
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = None
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10', s=40)
    ax.set_title('Agglomerative clusters (PCA 2D)')
    plt.tight_layout()
    return labels, sil, fig