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
                    parent_probs = [evidence[parent] for parent in node.parents]
                    if parent_probs == list(parent_values):
                        prob_sum += probability
                probabilities[node_name] = prob_sum
        return probabilities

# Define default input data
default_evidence = {"Fever": "High", "Cough": "Severe"}

# Define nodes and their probabilities
fever_node = Node("Fever", ["Low", "High"])
cough_node = Node("Cough", ["Mild", "Severe"])
corona_node = Node("Corona", ["Negative", "Positive"])

fever_node.add_parent(corona_node, {"Negative": 0.1, "Positive": 0.9})
cough_node.add_parent(corona_node, {"Negative": 0.2, "Positive": 0.8})

# Add nodes to the Bayesian network
network = BayesianNetwork()
network.add_node(fever_node)
network.add_node(cough_node)
network.add_node(corona_node)

# Perform inference with default input data
result = network.infer(default_evidence)
print("Probability of Corona infection:", result["Corona"])
