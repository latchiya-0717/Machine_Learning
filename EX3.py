import streamlit as st
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(1, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)
        
    def backward_propagation(self, X, y, learning_rate):

        output_error = y - self.predicted_output
        d_predicted_output = output_error * self.sigmoid_derivative(self.predicted_output)
        
        # Hidden layer
        hidden_error = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            self.forward_propagation(X)
            
            # Backward propagation
            self.backward_propagation(X, y, learning_rate)
                
    def predict(self, X):
        self.forward_propagation(X)
        return self.predicted_output

def main():
    st.title("BYTES BRIGADE")
    st.title('Backpropagation Neural Network with Streamlit')
    
    # Sample dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Neural network parameters
    input_size = 2
    hidden_size = st.slider('Hidden Layer Size', min_value=1, max_value=10, value=4)
    output_size = 1
    epochs = st.slider('Epochs', min_value=1000, max_value=10000, step=1000, value=5000)
    learning_rate = st.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1)
    
    # Initialize neural network
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    
    # Train the neural network
    nn.train(X, y, epochs=epochs, learning_rate=learning_rate)
    
    # Test the neural network
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = nn.predict(test_data)
    
    # Display predictions
    st.subheader('Predictions')
    st.write(predictions)

    

if __name__ == "__main__":
    main()
