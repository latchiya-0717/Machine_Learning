import streamlit as st
import numpy as np

# Function to calculate probabilities for Bayesian Network
def calculate_probabilities(X, y):
    total_samples = len(y)
    covid_positive = np.sum(y)
    covid_negative = total_samples - covid_positive

    prob_covid = covid_positive / total_samples
    prob_no_covid = covid_negative / total_samples

    prob_fever_given_covid = np.sum(X[y == 1, 0]) / covid_positive
    prob_fever_given_no_covid = np.sum(X[y == 0, 0]) / covid_negative

    prob_cough_given_covid = np.sum(X[y == 1, 1]) / covid_positive
    prob_cough_given_no_covid = np.sum(X[y == 0, 1]) / covid_negative

    prob_fatigue_given_covid = np.sum(X[y == 1, 2]) / covid_positive
    prob_fatigue_given_no_covid = np.sum(X[y == 0, 2]) / covid_negative

    return {
        'P(COVID)': prob_covid,
        'P(No COVID)': prob_no_covid,
        'P(Fever|COVID)': prob_fever_given_covid,
        'P(Fever|No COVID)': prob_fever_given_no_covid,
        'P(Cough|COVID)': prob_cough_given_covid,
        'P(Cough|No COVID)': prob_cough_given_no_covid,
        'P(Fatigue|COVID)': prob_fatigue_given_covid,
        'P(Fatigue|No COVID)': prob_fatigue_given_no_covid,
    }

# Function to diagnose COVID-19 using Bayesian network
def diagnose(fever, cough, fatigue, probabilities):
    p_covid = probabilities['P(COVID)'] * \
              (probabilities['P(Fever|COVID)'] if fever else 1 - probabilities['P(Fever|COVID)']) * \
              (probabilities['P(Cough|COVID)'] if cough else 1 - probabilities['P(Cough|COVID)']) * \
              (probabilities['P(Fatigue|COVID)'] if fatigue else 1 - probabilities['P(Fatigue|COVID)'])

    p_no_covid = probabilities['P(No COVID)'] * \
                 (probabilities['P(Fever|No COVID)'] if fever else 1 - probabilities['P(Fever|No COVID)']) * \
                 (probabilities['P(Cough|No COVID)'] if cough else 1 - probabilities['P(Cough|No COVID)']) * \
                 (probabilities['P(Fatigue|No COVID)'] if fatigue else 1 - probabilities['P(Fatigue|No COVID)'])

    return p_covid / (p_covid + p_no_covid)

# Bayesian Network Data
data = np.array([
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])

# Features: Fever, Cough, Fatigue
X = data[:, :-1]
# Target: COVID
y = data[:, -1]

probabilities = calculate_probabilities(X, y)

# Streamlit Interface
st.title('COVID-19 Bayesian Network Diagnosis')

st.header('Input Symptoms')
fever = st.selectbox('Fever', [0, 1])
cough = st.selectbox('Cough', [0, 1])
fatigue = st.selectbox('Fatigue', [0, 1])

if st.button('Diagnose'):
    prob_infection = diagnose(fever, cough, fatigue, probabilities)
    st.write(f"Probability of COVID-19 infection: {prob_infection:.2f}")
