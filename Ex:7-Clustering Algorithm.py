import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Function to load data from CSV file
def load_data(filename):
    return pd.read_csv(filename)

# Function to preprocess data
def preprocess_data(data):
    # Identify non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns.tolist()

    if non_numeric_columns:
        # One-hot encode non-numeric columns
        transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), non_numeric_columns)], 
                                         remainder='passthrough')
        data = transformer.fit_transform(data)
    
    return data

# Function to perform k-Means clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    return kmeans.labels_

# Function to perform EM clustering
def em_clustering(data, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(data)
    return gmm.predict(data)

# Main function
def main():
    st.title('Clustering Comparison')

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Preprocess the data
        data = preprocess_data(data)

        # Display the loaded and preprocessed data
        st.write("### Data Sample:")
        st.write(data)

        # Clustering
        num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10)
        algorithm = st.sidebar.selectbox("Select algorithm", ("k-Means", "EM"))

        if algorithm == "k-Means":
            labels = kmeans_clustering(data, num_clusters)
        elif algorithm == "EM":
            labels = em_clustering(data, num_clusters)

        # Visualize the results
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        ax.set_title("Clustering Results")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
