from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def run(X, params):
    n_clusters = int(params.get('n_clusters', 3))
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = None

    # 2D scatter via PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10', s=40)
    ax.set_title('KMeans clusters (PCA 2D)')
    plt.tight_layout()
    return labels, sil, fig