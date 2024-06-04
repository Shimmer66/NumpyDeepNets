# 激活函数及其导数
import pickle

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# 多分类
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# 均方误差损失函数及其导数
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# 损失函数及其导数
def cross_entropy_loss(y_true, y_pred):
    loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
    return loss


def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true


# MLP类定义
class MLPRegressor:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.z3  # 输出层
        return self.a3

    def backward(self, X, y):
        # 反向传播
        m = X.shape[0]  # 使用输入样本数
        y = y.reshape(-1, 1)
        # 计算输出层的梯度
        d_loss_a3 = self.a3 - y

        # 计算第三层的梯度  
        d_loss_W3 = np.dot(self.a2.T, d_loss_a3) / m
        d_loss_b3 = np.sum(d_loss_a3, axis=0, keepdims=True) / m

        # 计算第二层的梯度
        d_loss_z2 = np.dot(d_loss_a3, self.W3.T) * relu_derivative(self.z2)
        d_loss_W2 = np.dot(self.a1.T, d_loss_z2) / m
        d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True) / m

        # 计算第一层的梯度
        d_loss_z1 = np.dot(d_loss_z2, self.W2.T) * relu_derivative(self.z1)
        d_loss_W1 = np.dot(X.T, d_loss_z1) / m
        d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True) / m

        # 更新权重和偏置
        self.W3 -= self.learning_rate * d_loss_W3
        self.b3 -= self.learning_rate * d_loss_b3
        self.W2 -= self.learning_rate * d_loss_W2
        self.b2 -= self.learning_rate * d_loss_b2
        self.W1 -= self.learning_rate * d_loss_W1
        self.b1 -= self.learning_rate * d_loss_b1

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = mean_squared_error(y, output)
            self.backward(X, y)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        output = self.forward(X)
        return output

    def save_weights(self, file_path):
        weights = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3
        }
        with open(file_path, 'wb') as file:
            pickle.dump(weights, file)

    def load_weights(self, file_path):
        with open(file_path, 'rb') as file:
            weights = pickle.load(file)
            self.W1 = weights['W1']
            self.b1 = weights['b1']
            self.W2 = weights['W2']
            self.b2 = weights['b2']
            self.W3 = weights['W3']
            self.b3 = weights['b3']




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# 加载加州房价数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 标准化特征数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建MLP模型
mlp = MLPRegressor(input_size=X_train_scaled.shape[1], hidden_size1=64, hidden_size2=32, output_size=1,learning_rate=0.1)
#
# # 训练MLP模型
mlp.train(X_train_scaled, y_train, epochs=1000)
#
# 保存训练后的权重
mlp.save_weights('./weight/mlp_weights.pkl')