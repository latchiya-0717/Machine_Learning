import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Normalize data
def normalize_data(data):
    numeric_data = data.select_dtypes(include=[np.number])
    normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
    return normalized_data

# Initialize cluster centroids
def initialize_centroids(data, num_clusters):
    indices = np.random.choice(data.shape[0], size=num_clusters, replace=False)
    centroids = data[indices]
    return centroids

# E-step: Assign each data point to the nearest centroid
def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# M-step: Update centroids based on cluster assignments
def update_centroids(data, clusters, num_clusters):
    centroids = np.zeros((num_clusters, data.shape[1]))
    for i in range(num_clusters):
        centroids[i] = data[clusters == i].mean(axis=0)
    return centroids

# Expectation-Maximization (EM) algorithm
def em_algorithm(data, num_clusters, max_iterations=100):
    centroids = initialize_centroids(data, num_clusters)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, num_clusters)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters

# K-means algorithm
def kmeans_algorithm(data, num_clusters, max_iterations=100):
    centroids = initialize_centroids(data, num_clusters)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(num_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters

# Plot clusters
def plot_clusters(data, clusters, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue=clusters, palette='viridis', legend='full', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Streamlit web app
def main():
    st.title("Clustering Comparison: EM vs k-Means")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)

        # Display data
        st.subheader("Dataset")
        st.write(data)

        # Check if dataset has at least 2 columns
        if len(data.columns) >= 2:
            # Normalize data
            normalized_data = normalize_data(data.iloc[:, :2])

            # Select number of clusters
            num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

            # Clustering using EM algorithm
            em_labels = em_algorithm(normalized_data.values, num_clusters)
            plot_clusters(data.iloc[:, :2], em_labels, "EM Algorithm Clustering")

            # Clustering using k-means algorithm
            kmeans_labels = kmeans_algorithm(normalized_data.values, num_clusters)
            plot_clusters(data.iloc[:, :2], kmeans_labels, "k-Means Algorithm Clustering")
        else:
            st.error("Please ensure that your dataset has at least 2 columns.")

if __name__ == "__main__":
    main()
