import numpy as np
import pickle
import pandas as pd

# 激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 均方误差损失函数及其导数
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# MLP类定义
import numpy as np

import numpy as np


class MlpRegressor:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.000001):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.Z3  # Linear output for regression
        return self.A3

    def backward(self, X, y):
        m = y.shape[0]  # Number of samples

        # Calculate gradients for output layer
        dZ3 = self.A3 - y
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Calculate gradients for second hidden layer
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Calculate gradients for first hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Calculate loss (mean squared error)
            loss = np.mean((output - y) ** 2)

            # Backward pass and weight update
            self.backward(X, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)


from load_data import load_california_data

# 示例用法
if __name__ == "__main__":
    # 创建MLP模型
    mlp = MlpRegressor(input_size=7, hidden_size1=7, hidden_size2=7, output_size=1)
    train_X,  train_y = load_california_data('./data/housing.csv')
    # # 训练MLP模型
    mlp.train(train_X, train_y)
    #
    # # 保存训练后的权重
    # mlp.save_weights('./weights/mlpregressor_weights.pkl')
    # 评估模型
    train_accuracy = mlp.predict(train_X, train_y)


    # print(f'Train Accuracy: {train_accuracy * 100:.2f}%')

