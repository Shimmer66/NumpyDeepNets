import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 使用 UCI 机器学习库中的波士顿房价数据集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# 读取数据集
raw_df = pd.read_csv(data_url, delim_whitespace=True, names=column_names)

# 数据和目标分离
data = raw_df.drop(columns=["MEDV"]).values
target = raw_df["MEDV"].values

# 数据预处理
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 训练集和测试集分割
X_train, X_test, y_train, y_test = train_test_split(scaled_data, target, test_size=0.2, random_state=42)

class MLPRegressor:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01, reg_lambda=0.0, reg_alpha=0.0, decay_rate=0.0, optimizer='adam'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.parameters = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(1)
        self.parameters['W1'] = np.random.randn(self.input_size, self.hidden_layers[0]) * np.sqrt(2. / self.input_size)
        self.parameters['b1'] = np.zeros((1, self.hidden_layers[0]))
        for i in range(1, len(self.hidden_layers)):
            self.parameters['W' + str(i+1)] = np.random.randn(self.hidden_layers[i-1], self.hidden_layers[i]) * np.sqrt(2. / self.hidden_layers[i-1])
            self.parameters['b' + str(i+1)] = np.zeros((1, self.hidden_layers[i]))
        self.parameters['W' + str(len(self.hidden_layers) + 1)] = np.random.randn(self.hidden_layers[-1], self.output_size) * np.sqrt(2. / self.hidden_layers[-1])
        self.parameters['b' + str(len(self.hidden_layers) + 1)] = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward_propagation(self, X):
        cache = {}
        A = X
        cache['A0'] = A
        for i in range(1, len(self.hidden_layers) + 2):
            Z = np.dot(A, self.parameters['W' + str(i)]) + self.parameters['b' + str(i)]
            if i != len(self.hidden_layers) + 1:
                A = self.relu(Z)
            else:
                A = Z  # Output layer (no activation for regression)
            cache['A' + str(i)] = A
            cache['Z' + str(i)] = Z
        return A, cache

    def compute_cost(self, Y_hat, Y):
        m = Y.shape[0]
        cost = (1 / (2 * m)) * np.sum((Y_hat - Y) ** 2)
        return cost

    def backward_propagation(self, cache, X, Y):
        m = X.shape[0]
        grads = {}
        Y_hat = cache['A' + str(len(self.hidden_layers) + 1)]
        dA = Y_hat - Y

        for i in reversed(range(1, len(self.hidden_layers) + 2)):
            dZ = dA
            dW = (1 / m) * np.dot(cache['A' + str(i-1)].T, dZ)
            db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
            grads['dW' + str(i)] = dW
            grads['db' + str(i)] = db
            if i > 1:
                dA = np.dot(dZ, self.parameters['W' + str(i)].T) * self.relu_derivative(cache['Z' + str(i-1)])
        return grads

    def update_parameters(self, grads):
        for i in range(1, len(self.hidden_layers) + 2):
            self.parameters['W' + str(i)] -= self.learning_rate * grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= self.learning_rate * grads['db' + str(i)]

    def fit(self, X, Y, epochs=1000):
        Y = Y.reshape(-1, 1)
        for epoch in range(epochs):
            Y_hat, cache = self.forward_propagation(X)
            cost = self.compute_cost(Y_hat, Y)
            grads = self.backward_propagation(cache, X, Y)
            self.update_parameters(grads)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, cost: {cost}")

    def predict(self, X):
        Y_hat, _ = self.forward_propagation(X)
        return Y_hat

# 实例化MLPRegressor类
mlp_regressor = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size=1, activation='relu')

# 训练模型
mlp_regressor.fit(X_train, y_train, epochs=1000)

# 预测
predicted_values = mlp_regressor.predict(X_test)

# 输出结果
print("Predicted values shape:", predicted_values.shape)
print("Sample predicted values:", predicted_values[:5])  # 打印前5个预测值
print("Sample target values:", y_test[:5])  # 打印前5个真实目标值
