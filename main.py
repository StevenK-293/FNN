import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)
X = np.random.rand(100000, 10)  # 100,000 samples, 10 features each
y = (np.sum(X, axis=1) > 5).astype(int).reshape(-1, 1)  # Output is 1 if sum of inputs > 5, else 0

X = (X - X.mean(axis=0)) / X.std(axis=0)

split_ratio = 0.8
split_index = int(X.shape[0] * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

class FNN:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z = []
        self.a = [X]
        
        for i in range(len(self.weights) - 1):
            self.z.append(np.dot(self.a[i], self.weights[i]) + self.biases[i])
            self.a.append(self.relu(self.z[-1]))
        
        self.z.append(np.dot(self.a[-1], self.weights[-1]) + self.biases[-1])
        self.a.append(self.sigmoid(self.z[-1]))
        
        return self.a[-1]
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        epsilon = 1e-10  # Small value to avoid division by zero
        a_last = np.clip(self.a[-1], epsilon, 1 - epsilon)  # Clip values to avoid division by zero
        
        d_a = [-(y / a_last) + (1 - y) / (1 - a_last)]
        d_a[0] = d_a[0] * self.sigmoid_derivative(self.a[-1])
        
        for i in reversed(range(len(self.weights))):
            d_z = d_a[-1]
            d_w = np.dot(self.a[i].T, d_z) / m
            d_b = np.sum(d_z, axis=0, keepdims=True) / m
            
            self.weights[i] -= learning_rate * d_w
            self.biases[i] -= learning_rate * d_b
            
            if i != 0:
                d_a.append(np.dot(d_z, self.weights[i].T) * self.relu_derivative(self.a[i]))
    
    def train(self, X, y, epochs, learning_rate, batch_size):
        loss_history = []
        accuracy_history = []
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for i in tqdm(range(0, X.shape[0], batch_size), desc="Batches", leave=False):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                
                batch_loss = -np.mean(y_batch * np.log(output + 1e-10) + (1 - y_batch) * np.log(1 - output + 1e-10))
                batch_accuracy = np.mean((output > 0.5) == y_batch)
                
                epoch_loss += batch_loss * X_batch.shape[0]
                epoch_accuracy += batch_accuracy * X_batch.shape[0]
            
            epoch_loss /= X.shape[0]
            epoch_accuracy /= X.shape[0]
            
            loss_history.append(epoch_loss)
            accuracy_history.append(epoch_accuracy)
        
        return loss_history, accuracy_history
    
    def predict(self, X):
        return self.forward(X) > 0.5

nn = FNN(input_size=10, hidden_layer_sizes=[64, 32], output_size=1)
loss_history, accuracy_history = nn.train(X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32)

model_data = {
    'weights': nn.weights,
    'biases': nn.biases
}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

test_predictions = nn.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
