# neural network pratice for myself, its bad 

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from graphviz import Digraph
import pickle
import os

np.random.seed(42)

def gen_dataset(num_samples, num_features):
    X = np.random.rand(num_samples, num_features)
    y = (np.sum(X, axis=1) > (num_features / 2)).astype(int).reshape(-1, 1)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

def save_model(nn, file_path='model.pkl'):
    with open(file_path, 'wb') as file:
        pickle.dump((nn.weights, nn.biases), file)

def load_model(nn, file_path='model.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            nn.weights, nn.biases = pickle.load(file)
        return True
    return False

X, y = gen_dataset(1000000, 10) #1,000,000

# Splits the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

class FNN:
    def __init__(self, input_size, hidden_layer_sizes, output_size, l2_lambda=0.01, dropout_rate=0.5):
        self.weights = []
        self.biases = []
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.dropout_masks = []
        
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
    
    def forward(self, X, training=True):
        self.z = []
        self.a = [X]
        self.dropout_masks = []
        
        for i in range(len(self.weights) - 1):
            self.z.append(np.dot(self.a[i], self.weights[i]) + self.biases[i])
            activation = self.relu(self.z[-1])
            
            if training:
                dropout_mask = (np.random.rand(*activation.shape) > self.dropout_rate).astype(float)
                self.dropout_masks.append(dropout_mask)
                activation *= dropout_mask
                activation /= (1.0 - self.dropout_rate)
            
            self.a.append(activation)
        
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
            d_w = np.dot(self.a[i].T, d_z) / m + (self.l2_lambda * self.weights[i]) / m
            d_b = np.sum(d_z, axis=0, keepdims=True) / m
            
            self.weights[i] -= learning_rate * d_w
            self.biases[i] -= learning_rate * d_b
            
            if i != 0:
                d_a_prev = np.dot(d_z, self.weights[i].T) * self.relu_derivative(self.a[i])
                
                if len(self.dropout_masks) > 0:
                    d_a_prev *= self.dropout_masks[i - 1]
                
                d_a.append(d_a_prev)
    
    
    
    
    
    
    def train(self, X, y, X_val, y_val, epochs, learning_rate, batch_size, save_path='model.pkl'):
        loss_history = []
        accuracy_history = []
        val_accuracy_history = []
        
        best_val_accuracy = 0
        patience = 10
        patience_counter = 0
        initial_learning_rate = learning_rate
        
        for epoch in range(epochs):
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for i in tqdm(range(0, X.shape[0], batch_size), desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
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
            
            # calculate validation accuracy
            val_output = self.forward(X_val, training=False)
            val_accuracy = np.mean((val_output > 0.5) == y_val)
            val_accuracy_history.append(val_accuracy)
            
            # progress
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy * 100:.2f}% - Val Accuracy: {val_accuracy * 100:.2f}%")
            
            # early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                save_model(self, save_path)  # Saves the best model
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping at epoch:", epoch + 1)
                break
            
            #rate decay
            learning_rate = initial_learning_rate / (1 + epoch / epochs)
        
        return loss_history, accuracy_history, val_accuracy_history
    
    
    
    def predict(self, X):
        return self.forward(X, training=False) > 0.5
    
    def visualize(self, dpi=300):
        dot = Digraph()
        dot.attr(rankdir='LR', size='8,5')
        dot.attr(dpi=str(dpi))
        
        # Input layer
        with dot.subgraph(name='cluster_0') as c:
            c.attr(color='white')
            for i in range(len(self.a[0][0])):
                c.node(f'input_{i}', shape='circle', label=f'Input {i}')
            c.attr(label='Input Layer')
        
        # hidden layers
        for l in range(1, len(self.a) - 1):
            with dot.subgraph(name=f'cluster_{l}') as c:
                c.attr(color='white')
                for i in range(len(self.a[l][0])):
                    c.node(f'hidden_{l}_{i}', shape='circle', label=f'Hidden {l}_{i}')
                c.attr(label=f'Hidden Layer {l}')
        
        # Output layer
        with dot.subgraph(name=f'cluster_{len(self.a) - 1}') as c:
            c.attr(color='white')
            for i in range(len(self.a[-1][0])):
                c.node(f'output_{i}', shape='circle', label=f'Output {i}')
            c.attr(label='Output Layer')
        
        # connections
        for i in range(len(self.a[0][0])):
            for j in range(len(self.a[1][0])):
                dot.edge(f'input_{i}', f'hidden_1_{j}')
        
        for l in range(1, len(self.a) - 2):
            for i in range(len(self.a[l][0])):
                for j in range(len(self.a[l + 1][0])):
                    dot.edge(f'hidden_{l}_{i}', f'hidden_{l + 1}_{j}')
        
        for i in range(len(self.a[-2][0])):
            for j in range(len(self.a[-1][0])):
                dot.edge(f'hidden_{len(self.a) - 2}_{i}', f'output_{j}')
        
        return dot
    
    
    
    

def NN_Tree(input_size, hidden_layer_sizes, output_size):
    fig, ax = plt.subplots(figsize=(16, 16))
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    
    v_spacing = (1.0 / float(max(layer_sizes))) * 2
    h_spacing = (1.0 / float(len(layer_sizes) - 1)) * 2
    
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing, layer_top - j * v_spacing), v_spacing / 4,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if i == 0:
                ax.text(i * h_spacing - 0.1, layer_top - j * v_spacing, f'Input {j}', horizontalalignment='center', verticalalignment='center')
            elif i == len(layer_sizes) - 1:
                ax.text(i * h_spacing + 0.1, layer_top - j * v_spacing, f'Output {j}', horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(i * h_spacing, layer_top - j * v_spacing, f'Hidden {i}-{j}', horizontalalignment='center', verticalalignment='center')
    
    # draw edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2
        layer_top_b = v_spacing * (layer_size_b - 1) / 2
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing], c='k')
                ax.add_artist(line)
    
    ax.axis('off')
    
    plt.show()

nn = FNN(input_size=10, hidden_layer_sizes=[64, 32], output_size=1)

model_loaded = load_model(nn)

loss_history, accuracy_history, val_accuracy_history = nn.train(X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.01, batch_size=32)

# loss and accuracy history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Train Accuracy')
plt.plot(val_accuracy_history, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# model test set
test_predictions = nn.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape((1, -1)), cmap='viridis', aspect='auto')
    plt.title(f'Pred: {int(test_predictions[i][0])}\nTrue: {y_test[i][0]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

dot = nn.visualize(dpi=300)
dot.render('image', format='png', view=True)

NN_Tree(input_size=10, hidden_layer_sizes=[64, 32], output_size=1)
