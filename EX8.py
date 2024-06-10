import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def main():
    st.title("BYTES BRIGADE")
    st.title("K-Nearest Neighbors Classifier on Iris Dataset")
    
    # Load Iris dataset
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)
    
    # Train KNN model
    kn = KNeighborsClassifier(n_neighbors=1)
    kn.fit(X_train, y_train)
    
    # Display the dataset
    st.subheader("Iris Dataset")
    iris_data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    iris_data['target'] = dataset.target
    iris_data['target_name'] = iris_data['target'].apply(lambda x: dataset.target_names[x])
    st.write(iris_data)
    
    # Predict and display results
    st.subheader("Predictions on Test Data")
    results = []
    for i in range(len(X_test)):
        x = X_test[i]
        x_new = np.array([x])
        prediction = kn.predict(x_new)
        results.append({
            "Target": y_test[i],
            "Target Name": dataset["target_names"][y_test[i]],
            "Predicted": prediction[0],
            "Predicted Name": dataset["target_names"][prediction[0]]
        })
    
    results_df = pd.DataFrame(results)
    st.write(results_df)
    
    # Display model accuracy
    st.subheader("Model Accuracy")
    accuracy = kn.score(X_test, y_test)
    st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
