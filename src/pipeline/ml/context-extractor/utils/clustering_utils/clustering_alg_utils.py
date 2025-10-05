import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

class SubsequenceExtractor:
    @staticmethod
    def extract_fixed(series, window_size):
        subsequences = [
            series[j : j + window_size]
            for j in range(series.shape[0] - window_size + 1)
        ]
        return np.array(subsequences)

    @staticmethod
    def extract_variable(series, window_size, scaling_factor):
        subsequences, starts = [], []
        w_min, w_max = int(window_size / scaling_factor), int(
            window_size * scaling_factor
        )
        for w in range(w_min, w_max + 1):
            for i in range(series.shape[0] - w + 1):
                subseq = series[i : i + w]
                subseq = SubsequenceClusterer.uniform_scaling(subseq, window_size)
                subsequences.append(subseq)
                starts.append(i)
        return np.array(subsequences), starts


class SubsequenceScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, subsequences):
        n_subseq, w, f = subsequences.shape
        subsequences_flat = subsequences.reshape(n_subseq, w * f)
        return self.scaler.fit_transform(subsequences_flat)


class ClusterVisualizer:
    @staticmethod
    def plot_series_with_clusters(
        series,
        labels,
        window_size,
        n_clusters,
        feature_names=None,
        starts=None,
        title="Clustered Time Series",
    ):
        """
        Plots subsequence clusters overlay + all features in subplots (top/bottom).

        Args:
            series (ndarray): Time series data (timesteps x features).
            labels (list): Cluster labels for subsequences.
            window_size (int): Length of subsequences.
            n_clusters (int): Number of clusters.
            feature_names (list): List of feature names (len = n_features).
            starts (list): Starting indices of subsequences.
            title (str): Figure title.
        """
        n_features = series.shape[1]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        fig, axes = plt.subplots(
            2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )
        plt.subplots_adjust(hspace=0.3)

        # --------------------------
        # Top subplot: Overlay of first feature with cluster spans
        # --------------------------
        axes[0].plot(series[:, 0], color="black", linewidth=2, label=feature_names[0])
        for idx, cluster_id in enumerate(labels):
            start = idx if starts is None else starts[idx]
            axes[0].axvspan(
                start, start + window_size, color=colors[cluster_id % 10], alpha=0.005
            )
        axes[0].set_title(
            f"Overlay of {feature_names[0]}", fontsize=14, fontweight="bold"
        )
        axes[0].set_ylabel(feature_names[0])
        axes[0].legend()
        axes[0].grid(alpha=0.5)

        # --------------------------
        # Bottom subplot: All features with cluster spans
        # --------------------------
        for f_idx in range(n_features):
            axes[1].plot(series[:, f_idx], linewidth=1.5, label=feature_names[f_idx])
        for idx, cluster_id in enumerate(labels):
            start = idx if starts is None else starts[idx]
            axes[1].axvspan(
                start, start + window_size, color=colors[cluster_id % 10], alpha=0.005
            )
        axes[1].set_title(
            "All Features with Cluster Assignments", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Values")
        axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        axes[1].grid(alpha=0.5)

        # --------------------------
        # Global title
        # --------------------------
        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.show()

    @staticmethod
    def plot_pca(subsequences_scaled, labels, n_clusters, title=""):
        pca = PCA(n_components=2)
        subsequences_pca = pca.fit_transform(subsequences_scaled)
        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            plt.scatter(
                subsequences_pca[mask, 0],
                subsequences_pca[mask, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
            )
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.show()
        return subsequences_pca


class SubsequenceClusterer:
    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    @staticmethod
    def average_sequence(sequences):
        return np.mean(sequences, axis=0)

    @staticmethod
    def uniform_scaling(seq, target_len):
        idx = np.linspace(0, len(seq) - 1, target_len)
        return np.array([seq[int(round(i))] for i in idx])



class KMeansClusterer:
    def __init__(self, time_series_data, experiment_ids, random_state=0):
        self.data = time_series_data
        self.random_state = random_state
        self.experiment_ids = experiment_ids

    def cluster(
        self,
        experiment_id=2,
        window_size=10,
        n_clusters=6,
        feature_names=None,
        algorithm="kmeans",
    ):
        """
        Perform clustering on subsequences of a given time series and visualize results.

        Args:
            experiment_id (int/str): Experiment identifier.
            window_size (int): Length of subsequences.
            n_clusters (int): Number of clusters (ignored for DBSCAN).
            feature_names (list): Optional list of feature names for plots.
            algorithm (str): Clustering algorithm: "kmeans", "agglo", "dbscan", "kshape", "tskm".

        Returns:
            labels (ndarray): Cluster labels.
            model: Fitted clustering model.
            subsequences_pca (ndarray): PCA projection of subsequences.
        """
        # Map experiment_id to series index
        if experiment_id in self.experiment_ids:
            series_idx = self.experiment_ids.index(experiment_id)
        else:
            raise ValueError(
                f"Experiment ID '{experiment_id}' not found in experiment_ids."
            )

        series = self.data[series_idx]

        # Extract subsequences and scale
        subsequences = SubsequenceExtractor.extract_fixed(series, window_size)
        scaler = SubsequenceScaler()
        subsequences_scaled = scaler.fit_transform(subsequences)

        # -----------------------------
        # Choose clustering algorithm
        # -----------------------------
        algorithm = algorithm.lower()
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = model.fit_predict(subsequences_scaled)
        elif algorithm == "agglo":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(subsequences_scaled)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
            labels = model.fit_predict(subsequences_scaled)
        elif algorithm == "kshape":
            ts_data = to_time_series_dataset(subsequences)
            ts_data = TimeSeriesScalerMeanVariance().fit_transform(ts_data)
            model = KShape(n_clusters=n_clusters, random_state=self.random_state)
            labels = model.fit_predict(ts_data)
        elif algorithm == "tskm":
            ts_data = to_time_series_dataset(subsequences)
            ts_data = TimeSeriesScalerMeanVariance().fit_transform(ts_data)
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=self.random_state)
            labels = model.fit_predict(ts_data)
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from kmeans, agglo, dbscan, kshape, tskm.")

        # Visualization
        ClusterVisualizer.plot_series_with_clusters(
            series,
            labels,
            window_size,
            n_clusters,
            feature_names=feature_names,
            title=f"{algorithm.upper()} Clustering on Series {series_idx}",
        )

        subsequences_pca = ClusterVisualizer.plot_pca(
            subsequences_scaled,
            labels,
            n_clusters,
            title=f"{algorithm.upper()} PCA Projection (Series {series_idx})",
        )

        return labels, model, subsequences_pca


class KSelectionEvaluator:
    def __init__(self, time_series_data, experiment_ids, random_state=0):
        self.data = time_series_data
        self.experiment_ids = experiment_ids
        self.random_state = random_state

    def evaluate(self, experiment_id=2, window_size=10, k_range=range(2, 12)):
        """
        Evaluate optimal number of clusters using Elbow and Silhouette methods.

        Args:
            experiment_id (int/str): Experiment identifier (must exist in experiment_ids).
            window_size (int): Length of subsequences.
            k_range (range): Range of k values to test.

        Returns:
            inertias (list): List of inertia values for each k.
            scores (list): Silhouette scores for each k.
            best_k (int): Best number of clusters based on silhouette score.
        """
        # Validate experiment_id
        if experiment_id in self.experiment_ids:
            series_idx = self.experiment_ids.index(experiment_id)
        else:
            raise ValueError(
                f"Experiment ID '{experiment_id}' not found in experiment_ids."
            )

        # Pick series
        series = self.data[series_idx]

        # Extract subsequences and scale
        subsequences = SubsequenceExtractor.extract_fixed(series, window_size)
        scaler = SubsequenceScaler()
        subsequences_scaled = scaler.fit_transform(subsequences)

        inertias, scores = [], []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(subsequences_scaled)
            inertias.append(kmeans.inertia_)
            scores.append(silhouette_score(subsequences_scaled, labels))

        # Find best k by silhouette
        best_k = k_range[np.argmax(scores)]

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(wspace=0.3)

        # ------------------------
        # Elbow Method
        # ------------------------
        axes[0].plot(
            list(k_range), inertias, marker="o", linestyle="-", color="steelblue"
        )
        axes[0].set_title("Elbow Method", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inertia")
        axes[0].grid(True, linestyle="--", alpha=0.7)
        for i, val in enumerate(inertias):
            axes[0].annotate(
                f"{val:.0f}",
                (k_range[i], val),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        # ------------------------
        # Silhouette Method
        # ------------------------
        axes[1].plot(
            list(k_range), scores, marker="o", linestyle="-", color="darkorange"
        )
        axes[1].set_title("Silhouette Method", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
        axes[1].legend()
        for i, val in enumerate(scores):
            axes[1].annotate(
                f"{val:.2f}",
                (k_range[i], val),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        plt.suptitle(
            f"K Selection Evaluation (Experiment {experiment_id})",
            fontsize=16,
            fontweight="bold",
        )
        plt.show()

        return inertias, scores, best_k
