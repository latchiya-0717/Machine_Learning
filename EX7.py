import streamlit as st
import numpy as np

# Clustering Data
cluster_data = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
])

# EM Algorithm
def initialize_clusters(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def e_step(X, means):
    responsibilities = np.zeros((len(X), len(means)))
    for i, x in enumerate(X):
        for j, mean in enumerate(means):
            responsibilities[i, j] = np.exp(-0.5 * np.linalg.norm(x - mean)**2)
        responsibilities[i] /= responsibilities[i].sum()
    return responsibilities

def m_step(X, responsibilities):
    means = np.zeros((responsibilities.shape[1], X.shape[1]))
    for j in range(responsibilities.shape[1]):
        weighted_sum = np.dot(responsibilities[:, j], X)
        sum_weights = responsibilities[:, j].sum()
        means[j] = weighted_sum / sum_weights
    return means

def em_algorithm(X, k, max_iters=100):
    means = initialize_clusters(X, k)
    for _ in range(max_iters):
        responsibilities = e_step(X, means)
        means = m_step(X, responsibilities)
    return means, responsibilities.argmax(axis=1)

# k-Means Algorithm
def kmeans(X, k, max_iters=100):
    centroids = initialize_clusters(X, k)
    for _ in range(max_iters):
        clusters = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = np.linalg.norm(x - centroids, axis=1)
            clusters[i] = np.argmin(distances)
        for j in range(k):
            points = X[clusters == j]
            if len(points) > 0:
                centroids[j] = points.mean(axis=0)
    return centroids, clusters

# Streamlit Interface
st.title('Clustering Using EM Algorithm and k-Means')

st.header('Clustering Data')
st.write(cluster_data)

num_clusters = st.slider('Number of clusters', 1, 5, 2)

if st.button('Cluster Data'):
    em_means, em_clusters = em_algorithm(cluster_data, num_clusters)
    kmeans_centroids, kmeans_clusters = kmeans(cluster_data, num_clusters)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EM Algorithm Results")
        st.write("Means:")
        st.write(em_means)
        st.write("Clusters:")
        st.write(em_clusters)

    with col2:
        st.subheader("k-Means Algorithm Results")
        st.write("Centroids:")
        st.write(kmeans_centroids)
        st.write("Clusters:")
        st.write(kmeans_clusters)

    st.header("Comparison of EM and k-Means clustering")
    col1, col2 = st.columns(2)

    with col1:
        st.write("EM clusters:")
        st.write(em_clusters)

    with col2:
        st.write("k-Means clusters:")
        st.write(kmeans_clusters)
