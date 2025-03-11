import numpy as np
 # Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Input vector (2 features)
x = np.array([[0.5], [0.8]])

# Output
y_true = np.array([[1]])

# Initialize weights
n_input = 2
n_hidden = 2
n_output = 1

# Limits
limit1 = np.sqrt(1 / n_input)
W1 = np.random.uniform(-limit1, limit1, (n_hidden, n_input))
b1 = np.zeros((n_hidden, 1))

limit2 = np.sqrt(1 / n_hidden)
W2 = np.random.uniform(-limit2, limit2, (n_output, n_hidden))
b2 = np.zeros((n_output, 1))

# Forward pass
z1 = W1 @ x + b1
a1 = sigmoid(z1)

z2 = W2 @ a1 + b2
a2 = sigmoid(z2)

#Compute loss with BCE
loss = -(y_true * np.log(a2) + (1 - y_true) * np.log(1 - a2))
print(f"Loss: {loss[0][0]:.4f}")

# Backprop
dz2 = a2 - y_true
dW2 = dz2 @ a1.T
db2 = dz2

dz1 = (W2.T @ dz2) * sigmoid_derivative(z1)
dW1 = dz1 @ x.T
db1 = dz1

# Update gradient descent
lr = 0.1  # Learning rate

W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2

# Updated weights and biases
print("\nUpdated Parameters:")
print("W1:\n", W1)
print("b1:\n", b1)
print("W2:\n", W2)
print("b2:\n", b2)