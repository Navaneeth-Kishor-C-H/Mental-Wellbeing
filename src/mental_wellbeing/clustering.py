"""Unsupervised cluster benchmarking and deployment-friendly cluster assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


class ClusterAssigner:
    """Stores centroids, allowing every selected algorithm to label new students."""
    def __init__(self, scaler: StandardScaler, centroids: np.ndarray, algorithm: str):
        self.scaler, self.centroids, self.algorithm = scaler, centroids, algorithm

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.scaler.transform(frame)
        distances = ((values[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)


def benchmark_clusters(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, ClusterAssigner, np.ndarray]:
    # Hierarchical clustering is quadratic, so benchmark a reproducible sample.
    # The selected centroid assigner then labels every student in the full dataset.
    sample = frame[feature_names].sample(n=min(len(frame), 5000), random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(sample)
    values = scaler.transform(sample)
    candidates = []
    fitted = []
    for clusters in range(2, 7):
        algorithms = {
            "K-Means": KMeans(n_clusters=clusters, n_init=20, random_state=RANDOM_STATE),
            "Agglomerative": AgglomerativeClustering(n_clusters=clusters, linkage="ward"),
            "Gaussian Mixture": GaussianMixture(n_components=clusters, random_state=RANDOM_STATE, n_init=3),
        }
        for name, estimator in algorithms.items():
            labels = estimator.fit_predict(values)
            candidates.append({"algorithm": name, "n_clusters": clusters, "silhouette": silhouette_score(values, labels), "davies_bouldin": davies_bouldin_score(values, labels), "calinski_harabasz": calinski_harabasz_score(values, labels)})
            fitted.append((name, clusters, labels))
    scores = pd.DataFrame(candidates)
    # Equal-weight rank aggregation: high silhouette/CH and low Davies-Bouldin are desirable.
    scores["selection_score"] = (scores["silhouette"].rank(ascending=False, method="min") + scores["davies_bouldin"].rank(ascending=True, method="min") + scores["calinski_harabasz"].rank(ascending=False, method="min")) / 3
    scores = scores.sort_values(["selection_score", "silhouette"], ascending=[True, False]).reset_index(drop=True)
    winner = scores.iloc[0]
    _, _, labels = next(item for item in fitted if item[0] == winner.algorithm and item[1] == winner.n_clusters)
    centroids = np.vstack([values[labels == cluster].mean(axis=0) for cluster in range(int(winner.n_clusters))])
    return scores, ClusterAssigner(scaler, centroids, str(winner.algorithm)), labels
