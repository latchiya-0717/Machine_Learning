import streamlit as st

# Define nodes in the Bayesian network
class Node:
    def __init__(self, name, states):
        self.name = name
        self.states = states
        self.parents = []
        self.probabilities = {}

    def add_parent(self, parent, probabilities):
        self.parents.append(parent)
        self.probabilities[parent.name] = probabilities

    def get_probability(self, parent_values):
        return self.probabilities[parent_values]

# Define the Bayesian network structure
class BayesianNetwork:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def infer(self, evidence):
        probabilities = {}
        for node_name, node in self.nodes.items():
            if node_name not in evidence:
                prob_sum = 0
                for parent_values, probability in node.probabilities.items():
                    parent_probs = tuple([evidence[parent] for parent in node.parents])
                    if parent_probs == parent_values:
                        prob_sum += probability
                probabilities[node_name] = prob_sum
        return probabilities

# Function to run inference and display results
def run_inference(network, evidence):
    result = network.infer(evidence)
    return result["Corona"]

# Streamlit web application
def main():
    st.title("Corona Infection Probability Calculator")
    st.write("Enter symptoms below to calculate the probability of a Corona infection.")

    # Create input fields for symptoms
    fever = st.selectbox("Fever", ["Low", "High"], index=1)
    cough = st.selectbox("Cough", ["Mild", "Severe"], index=1)

    # Define default evidence
    default_evidence = {"Fever": fever, "Cough": cough}

    # Define nodes and their probabilities
    fever_node = Node("Fever", ["Low", "High"])
    cough_node = Node("Cough", ["Mild", "Severe"])
    corona_node = Node("Corona", ["Negative", "Positive"])

    fever_node.add_parent(corona_node, {("Negative",): 0.1, ("Positive",): 0.9})
    cough_node.add_parent(corona_node, {("Negative",): 0.2, ("Positive",): 0.8})

    # Add nodes to the Bayesian network
    network = BayesianNetwork()
    network.add_node(fever_node)
    network.add_node(cough_node)
    network.add_node(corona_node)

    # Perform inference with user input
    probability = run_inference(network, default_evidence)

    # Display result
    st.write(f"Probability of Corona infection: {probability}")

if __name__ == "__main__":
    main()
