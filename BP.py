import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros(output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta) * self.learning_rate
        self.bias_hidden_output += np.sum(self.output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(self.hidden_delta) * self.learning_rate
        self.bias_input_hidden += np.sum(self.hidden_delta) * self.learning_rate

# Generate synthetic data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
def main():
    st.title("Simple Neural Network with Streamlit")

    # User inputs
    input_size = st.sidebar.slider("Input size", min_value=1, max_value=10, value=2)
    hidden_size = st.sidebar.slider("Hidden layer size", min_value=1, max_value=10, value=4)
    learning_rate = st.sidebar.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1)
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=1000, value=100)
    
    # Train the neural network
    nn = NeuralNetwork(input_size, hidden_size, 1, learning_rate)
    st.write("Training the neural network...")
    for epoch in range(epochs):
        output = nn.forward(X_train)
        nn.backward(X_train, y_train.reshape(-1, 1), output)
        if epoch % 10 == 0:
            loss = np.mean(np.square(y_train.reshape(-1, 1) - output))
            st.write(f"Epoch: {epoch}, Loss: {loss}")
    
    # Test the neural network
    output_test = nn.forward(X_test)
    y_pred = np.round(output_test).flatten()
    y_pred[np.isnan(y_pred)] = 0  # Set NaN values to 0
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy on test set: {accuracy}")
    
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-2, 2, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(X_grid)
    Z = np.round(Z).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolors='k')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
