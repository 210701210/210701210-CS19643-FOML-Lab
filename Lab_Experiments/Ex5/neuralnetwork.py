import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward propagation through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output, learning_rate):
        # Backpropagation through the network
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta) * learning_rate
        self.bias_hidden_output += np.sum(self.output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(self.hidden_delta) * learning_rate
        self.bias_input_hidden += np.sum(self.hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

if __name__ == "__main__":
    # Example usage
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
    y = np.array([[0], [1], [1], [0]])             # Output

    # Initialize neural network
    input_size = 2
    hidden_size = 4
    output_size = 1
    neural_network = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    epochs = 10000
    learning_rate = 0.1
    neural_network.train(X, y, epochs, learning_rate)

    # Test the trained network
    print("Final predictions:")
    print(neural_network.forward(X))
