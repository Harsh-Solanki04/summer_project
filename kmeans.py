import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples,silhouette_score


def load_data(file_path="C:\\Users\\harsh\\OneDrive\\Desktop\\Summer Project-KMeans\\Country_data.csv"):
    """Load the dataset from the specified file."""
    data = pd.read_csv(file_path)
    return data


def standardize_data(data):
    """Standardize the dataset features."""
    features = data.drop(columns=['country'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
    return scaled_data


def attribute_wise_clustering(data, k):
    """Cluster each attribute independently and generate pattern strings."""
    n_attributes = data.shape[1]
    pattern_strings = np.zeros((data.shape[0], n_attributes), dtype=int)

    for i, column in enumerate(data.columns):
        attribute_values = data[column].values.reshape(-1, 1)
        
        # Generate percentiles to ensure k initial centers
        percentiles = np.linspace(0, 100, k + 1)
        initial_centers = np.percentile(attribute_values, percentiles[1:-1]).reshape(-1, 1)
        
        # Handle cases where fewer centers are generated than k
        if len(initial_centers) != k:
            unique_vals = np.unique(attribute_values.flatten())
            if len(unique_vals) >= k:
                initial_centers = np.random.choice(unique_vals, k).reshape(-1, 1)
            else:
                initial_centers = np.random.choice(attribute_values.flatten(), k).reshape(-1, 1)
        
        # Perform K-means clustering on this attribute
        labels, _ = kmeans_clustering(attribute_values, k, initial_centers)
        pattern_strings[:, i] = labels  # Store cluster labels for the attribute

    return pattern_strings


def density_based_condensation(data, pattern_strings, k):
    """Merge pattern string clusters into the desired number of clusters."""
    unique_patterns, counts = np.unique(pattern_strings, axis=0, return_counts=True)
    density = counts / np.sum(counts)  # Density of each pattern string

    # Sort by density in descending order
    sorted_indices = np.argsort(density)[::-1]
    condensed_clusters = []

    # Select the top-k dense patterns
    for idx in sorted_indices:
        if len(condensed_clusters) >= k:
            break
        condensed_clusters.append(unique_patterns[idx])
    
    # Assign points to condensed clusters and compute final centers
    final_centers = []
    for cluster in condensed_clusters:
        mask = np.all(pattern_strings == cluster, axis=1)
        cluster_points = data[mask]
        if len(cluster_points) > 0:
            final_centers.append(cluster_points.mean(axis=0))
        else:
            # Handle edge case: Assign a random center
            final_centers.append(data[np.random.choice(data.shape[0])])
    
    return np.array(final_centers)


def kmeans_clustering(data, k, init_centers, max_iters=100, tol=1e-4):
    """Run K-means algorithm with handling for empty clusters."""
    centers = init_centers
    for _ in range(max_iters):
        distances = cdist(data, centers)
        labels = np.argmin(distances, axis=1)
        
        # Update centers with mean of assigned points
        new_centers = []
        for i in range(k):
            cluster_points = data[labels == i]
            if cluster_points.size == 0:
                # Handle empty cluster: Assign a random point as the center
                new_centers.append(data[np.random.choice(data.shape[0])])
            else:
                new_centers.append(cluster_points.mean(axis=0))
        
        new_centers = np.array(new_centers)

        # Convergence check
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers

    return labels, centers


def visualise_clusters(data, labels, centers, pca_data):
    """Visualize the clusters in 2D using PCA."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette="Set2", s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')
    plt.title("Clusters and Centers")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Load and preprocess data
    country_data = load_data()
    scaled_data = standardize_data(country_data)
    scaled_data_array = scaled_data.to_numpy()

    # Step 1: Perform attribute-wise clustering
    optimal_k = 4  # Assume the number of clusters is known
    pattern_strings = attribute_wise_clustering(scaled_data, optimal_k)

    # Step 2: Perform density-based merging (DBMSDC)
    initial_centers = density_based_condensation(scaled_data_array, pattern_strings, optimal_k)

    # Step 3: Final K-means clustering using CCIA centers
    labels, centers = kmeans_clustering(scaled_data_array, optimal_k, initial_centers)

    # Add PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data_array)
    pca_centers = pca.transform(centers)

    # Visualize clusters
    visualise_clusters(scaled_data, labels, pca_centers, pca_data)
