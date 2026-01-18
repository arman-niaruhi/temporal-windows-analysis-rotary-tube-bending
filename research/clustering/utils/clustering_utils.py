import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans

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
    one_series = time_series_data[series_idx]  

    subsequences = [one_series[j:j+window_size] 
                    for j in range(one_series.shape[0] - window_size + 1)]
    subsequences = np.array(subsequences)  

    n_subseq, w, f = subsequences.shape
    subsequences_flat = subsequences.reshape(n_subseq, w*f)

    scaler = StandardScaler()
    subsequences_scaled = scaler.fit_transform(subsequences_flat)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(subsequences_scaled)

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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def average_sequence(sequences):
    return np.mean(sequences, axis=0)

def uniform_scaling(seq, target_len):
    idx = np.linspace(0, len(seq) - 1, target_len)
    return np.array([seq[int(round(i))] for i in idx])

def cluster_time_series_subsequences_ssts(time_series_data, series_idx=0, window_size=10, scaling_factor=1.5, random_state=0):
    """
    Selective Subsequence Time Series (SSTS) clustering for a multivariate time series.

    Parameters
    ----------
    time_series_data : np.ndarray
        Array of shape (num_series, timesteps, num_features).
    series_idx : int
        Index of the time series to use.
    window_size : int
        Base length of subsequences (sliding window).
    scaling_factor : float
        Factor to vary subsequence length (min = w/f, max = w*f).
    random_state : int
        Random seed for reproducibility.
    """
    np.random.seed(random_state)

    one_series = time_series_data[series_idx]  

    w_min, w_max = int(window_size / scaling_factor), int(window_size * scaling_factor)
    subsequences = []
    starts = []
    for w in range(w_min, w_max + 1):
        for i in range(one_series.shape[0] - w + 1):
            subseq = one_series[i:i+w]
            subseq = uniform_scaling(subseq, window_size)  
            subsequences.append(subseq)
            starts.append(i)

    subsequences = np.array(subsequences)  
    n_subseq, w, f = subsequences.shape
    subsequences_flat = subsequences.reshape(n_subseq, w*f)

    scaler = StandardScaler()
    subsequences_scaled = scaler.fit_transform(subsequences_flat)

    clusters = []
    centers = []
    labels = np.full(n_subseq, -1)

    for i in range(n_subseq):
        subseq = subsequences_scaled[i]
        if not centers:
            centers.append(subseq)
            clusters.append([i])
            labels[i] = 0
        else:
            dists = [euclidean_distance(subseq, c) for c in centers]
            j = np.argmin(dists)
            if dists[j] < 1.0:  
                clusters[j].append(i)
                centers[j] = average_sequence(subsequences_scaled[clusters[j]])
                labels[i] = j
            else:
                centers.append(subseq)
                clusters.append([i])
                labels[i] = len(centers) - 1

    n_clusters = len(centers)

    plt.figure(figsize=(15, 4))
    plt.plot(one_series[:, 0], color="black", linewidth=2, label="Feature 0 (example)")

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for idx, cluster_id in enumerate(labels):
        start = starts[idx]
        plt.axvspan(start, start + window_size, color=colors[cluster_id % 10], alpha=0.3)

    plt.title(f"SSTS Cluster overlay on Feature 0 (Series {series_idx})")
    plt.xlabel("Time")
    plt.ylabel("Feature 0 value")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 6))
    for f_idx in range(one_series.shape[1]):
        plt.plot(one_series[:, f_idx], linewidth=1.5, label=f"Feature {f_idx}")

    for idx, cluster_id in enumerate(labels):
        start = starts[idx]
        plt.axvspan(start, start + window_size, color=colors[cluster_id % 10], alpha=0.15)

    plt.title(f"SSTS Clustered subsequences (Series {series_idx})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    pca = PCA(n_components=2)
    subsequences_pca = pca.fit_transform(subsequences_scaled)

    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(subsequences_pca[mask, 0], subsequences_pca[mask, 1],
                    label=f"Cluster {cluster_id}", alpha=0.6)
    plt.title(f"SSTS PCA of clustered subsequences (Series {series_idx})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

    return labels, centers, subsequences_pca





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
    one_series = time_series_data[series_idx]  

    subsequences = [one_series[j:j+window_size] 
                    for j in range(one_series.shape[0] - window_size + 1)]
    subsequences = np.array(subsequences)  

    n_subseq, w, f = subsequences.shape
    subsequences_flat = subsequences.reshape(n_subseq, w*f)

    scaler = StandardScaler()
    subsequences_scaled = scaler.fit_transform(subsequences_flat)
    
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
