import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cluster_time_series_subsequences(time_series_data, series_idx=0, window_size=10, n_clusters=6, random_state=0):
    """
    Cluster subsequences of a selected multivariate time series, visualize clusters,
    and show PCA of clustered subsequences.

    Parameters
    ----------
    time_series_data : np.ndarray
        Array of shape (num_series, timesteps, num_features).
    series_idx : int
        Index of the time series to use.
    window_size : int
        Length of subsequences (sliding window).
    n_clusters : int
        Number of clusters for KMeans.
    random_state : int
        Random state for reproducibility.
    """
    # Pick one series
    one_series = time_series_data[series_idx]  # (timesteps, num_features)

    # Extract subsequences
    subsequences = [one_series[j:j+window_size] 
                    for j in range(one_series.shape[0] - window_size + 1)]
    subsequences = np.array(subsequences)  # (n_subseq, window_size, features)

    # Flatten subsequences for clustering
    n_subseq, w, f = subsequences.shape
    subsequences_flat = subsequences.reshape(n_subseq, w*f)

    # Standardize
    scaler = StandardScaler()
    subsequences_scaled = scaler.fit_transform(subsequences_flat)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(subsequences_scaled)

    # -----------------------
    # Plot feature 0 with cluster mask
    # -----------------------
    plt.figure(figsize=(15, 4))
    plt.plot(one_series[:, 0], color="black", linewidth=2, label="Feature 0 (example)")

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i, cluster_id in enumerate(labels):
        plt.axvspan(i, i + window_size, color=colors[cluster_id], alpha=0.3)

    plt.title(f"Cluster overlay on Feature 0 (Series {series_idx})")
    plt.xlabel("Time")
    plt.ylabel("Feature 0 value")
    plt.legend()
    plt.show()

    # -----------------------
    # Plot all features with cluster mask
    # -----------------------
    plt.figure(figsize=(15, 6))
    for f_idx in range(one_series.shape[1]):
        plt.plot(one_series[:, f_idx], linewidth=1.5, label=f"Feature {f_idx}")

    for i, cluster_id in enumerate(labels):
        plt.axvspan(i, i + window_size, color=colors[cluster_id], alpha=0.15)

    plt.title(f"Clustered subsequences (Series {series_idx})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # -----------------------
    # PCA of clustered subsequences
    # -----------------------
    pca = PCA(n_components=2)
    subsequences_pca = pca.fit_transform(subsequences_scaled)

    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(subsequences_pca[mask, 0], subsequences_pca[mask, 1], 
                    label=f"Cluster {cluster_id}", alpha=0.6)
    plt.title(f"PCA of clustered subsequences (Series {series_idx})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

    return labels, kmeans, subsequences_pca




def plot_k_selection(time_series_data, series_idx, window_size, k_range=range(2, 12)):
    """
    Plot Elbow method and Silhouette scores to select optimal number of clusters.
    
    Parameters
    ----------
    subsequences_scaled : np.ndarray
        Standardized subsequences, shape (n_subseq, features)
    k_range : range
        Range of k values to test
    """
    # Pick one series
    one_series = time_series_data[series_idx]  # (timesteps, num_features)

    # Extract subsequences
    subsequences = [one_series[j:j+window_size] 
                    for j in range(one_series.shape[0] - window_size + 1)]
    subsequences = np.array(subsequences)  # (n_subseq, window_size, features)

    # Flatten subsequences for clustering
    n_subseq, w, f = subsequences.shape
    subsequences_flat = subsequences.reshape(n_subseq, w*f)

    # Standardize
    scaler = StandardScaler()
    subsequences_scaled = scaler.fit_transform(subsequences_flat)
    
    # -----------------------
    # Elbow method
    # -----------------------
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(subsequences_scaled)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(6, 4))
    plt.plot(list(k_range), inertias, marker='o', linestyle='-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()
    
    # -----------------------
    # Silhouette method
    # -----------------------
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(subsequences_scaled)
        score = silhouette_score(subsequences_scaled, labels)
        scores.append(score)
    
    plt.figure(figsize=(6, 4))
    plt.plot(list(k_range), scores, marker='o', linestyle='-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method for Optimal k")
    plt.grid(True)
    plt.show()
    
    return inertias, scores
