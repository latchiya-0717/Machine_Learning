import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data(dataset_file):
    data = pd.read_csv(dataset_file)
    return data

# Simulate symptoms based on confirmed cases
def simulate_symptoms(data):
    # Assuming a simple rule: If confirmed cases > 0, then fever and cough are present
    data['Fever'] = data['Confirmed'].apply(lambda x: 'Yes' if x > 0 else 'No')
    data['Cough'] = data['Confirmed'].apply(lambda x: 'Yes' if x > 0 else 'No')
    # Assuming Fatigue and Breathing Difficulty are random
    data['Fatigue'] = ['Yes' if i % 2 == 0 else 'No' for i in range(len(data))]
    data['Breathing Difficulty'] = ['Yes' if i % 3 == 0 else 'No' for i in range(len(data))]
    return data

# Calculate probabilities
def calculate_probabilities(data, symptoms, test_result):
    total_cases = len(data)
    symptomatic_cases = len(data[(data['Fever'] == symptoms['Fever']) & 
                                 (data['Cough'] == symptoms['Cough']) & 
                                 (data['Fatigue'] == symptoms['Fatigue']) &
                                 (data['Breathing Difficulty'] == symptoms['Breathing Difficulty'])])
    positive_test_cases = len(data[(data['Confirmed'] > 0)])
    
    # Calculate probabilities
    p_symptomatic_given_positive = len(data[(data['Confirmed'] > 0) & 
                                            (data['Fever'] == symptoms['Fever']) & 
                                            (data['Cough'] == symptoms['Cough']) & 
                                            (data['Fatigue'] == symptoms['Fatigue']) &
                                            (data['Breathing Difficulty'] == symptoms['Breathing Difficulty'])]) / positive_test_cases
    
    p_positive = positive_test_cases / total_cases
    p_symptomatic = symptomatic_cases / total_cases
    
    # Bayes' Theorem
    p_positive_given_symptomatic = (p_symptomatic_given_positive * p_positive) / p_symptomatic
    
    return p_positive_given_symptomatic

# Streamlit web app
def main():
    st.title("COVID-19 Diagnosis using Bayesian Network")

    # Ask for dataset file
    dataset_file = st.file_uploader("Upload dataset file", type=["csv"])

    if dataset_file is not None:
        # Load data
        data = load_data(dataset_file)

        # Simulate symptoms
        data = simulate_symptoms(data)

        # Show the dataset
        st.subheader("Dataset")
        st.write(data)

        # Inputs
        st.sidebar.title("Enter Symptoms")
        fever = st.sidebar.radio("Fever", ['Yes', 'No'])
        cough = st.sidebar.radio("Cough", ['Yes', 'No'])
        fatigue = st.sidebar.radio("Fatigue", ['Yes', 'No'])
        breathing_difficulty = st.sidebar.radio("Breathing Difficulty", ['Yes', 'No'])

        # Predict
        symptoms = {'Fever': fever, 'Cough': cough, 'Fatigue': fatigue, 'Breathing Difficulty': breathing_difficulty}
        test_result = 'Positive'
        probability = calculate_probabilities(data, symptoms, test_result)

        # Show prediction
        st.subheader("Probability of Positive Test Result given Symptoms")
        st.write(probability)

if __name__ == "__main__":
    main()
