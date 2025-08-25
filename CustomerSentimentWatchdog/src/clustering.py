import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_texts(texts, k=5, max_features=3000):
    texts = [t if isinstance(t, str) else '' for t in texts]
    vec = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1,2))
    X = vec.fit_transform(texts)
    if X.shape[0] == 0:
        return np.array([]), []
    k = min(k, max(2, min(10, X.shape[0])))
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X)
    # top terms per cluster
    terms = vec.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_terms = []
    for i in range(k):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        cluster_terms.append(top_terms)
    return labels, cluster_terms
