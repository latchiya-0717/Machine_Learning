import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load example dataset (replace with your actual dataset)
data = pd.read_csv('corona_dataset.csv')

# Define Bayesian Network structure
model = BayesianModel([('Fever', 'COVID19'), 
                        ('Cough', 'COVID19'),
                        ('Difficulty_breathing', 'COVID19'),
                        ('COVID19', 'Test_result')])

# Estimate parameters using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)

# Function to diagnose COVID-19 based on symptoms
def diagnose_covid19(fever, cough, difficulty_breathing):
    evidence = {'Fever': fever,
                'Cough': cough,
                'Difficulty_breathing': difficulty_breathing}
    result = inference.query(variables=['COVID19'], evidence=evidence)
    return result.values

# Example usage
fever = 'High'
cough = 'Yes'
difficulty_breathing = 'Yes'
probability_covid19 = diagnose_covid19(fever, cough, difficulty_breathing)
print("Probability of COVID-19:", probability_covid19)
