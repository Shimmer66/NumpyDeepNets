# 该代码有问题，后续优化
#
#
import pickle

import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    """
    ReLU激活函数
    :param x: 输入数组
    :return: 对应每个元素的ReLU值
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU激活函数的导数
    :param x: 输入数组
    :return: 对应每个元素的ReLU导数值
    """
    return np.where(x > 0, 1, 0)


def softmax(x):
    """
    Softmax激活函数
    :param x: 输入数组
    :return: 对应每个元素的Softmax概率值
    """
    # 减去每行的最大值以提高数值稳定性，防止溢出
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# 损失函数及其导数

def cross_entropy_loss(y_true, y_pred):
    """
    交叉熵损失函数
    :param y_true: 真实标签的one-hot编码
    :param y_pred: 模型预测的概率分布
    :return: 交叉熵损失值
    """
    # 加上一个小值1e-15以防止计算对数时出现log(0)
    loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
    return loss


def cross_entropy_loss_derivative(y_true, y_pred):
    """
    交叉熵损失函数的导数
    :param y_true: 真实标签的one-hot编码
    :param y_pred: 模型预测的概率分布
    :return: 交叉熵损失对预测值的梯度
    """
    return y_pred - y_true


def mean_squared_error(y_true, y_pred):
    """
    均方误差损失函数
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 均方误差损失值
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_error_derivative(y_true, y_pred):
    """
    均方误差损失函数的导数
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 均方误差损失对预测值的梯度
    """
    return np.mean(2 * (y_pred - y_true))


# class DenseLayer:
#     def __init__(self, input_size, output_size, activation, activation_derivative, l2_reg=0.0):
#         """
#         初始化全连接层
#         :param input_size: 输入大小
#         :param output_size: 输出大小
#         :param activation: 激活函数
#         :param activation_derivative: 激活函数的导数
#         """
#         self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)  # He initialization
#         self.b = np.zeros((1, output_size))
#         self.activation = activation
#         self.activation_derivative = activation_derivative
#         self.l2_reg = l2_reg
#
#         # 批量归一化参数
#         self.gamma = np.ones((1, output_size))
#         self.beta = np.zeros((1, output_size))
#
#     def forward(self, X, is_training=False):
#         """
#         前向传播
#         :param X: 输入数据
#         :param is_training: 是否为训练模式
#         :return: 激活值
#         """
#         self.z = np.dot(X, self.W) + self.b
#         # 批量归一化
#         if is_training:
#             self.mean = np.mean(self.z, axis=0, keepdims=True)
#             self.var = np.var(self.z, axis=0, keepdims=True)
#             self.z_norm = (self.z - self.mean) / np.sqrt(self.var + 1e-8)
#         else:
#             self.z_norm = (self.z - self.running_mean) / np.sqrt(self.running_var + 1e-8)
#
#         self.a = self.gamma * self.z_norm + self.beta
#         self.a = self.activation(self.a)
#         return self.a
#
#     def backward(self, dA, A_prev, learning_rate):
#         m = A_prev.shape[0]
#
#         # 批量归一化反向传播
#         dA_norm = dA * self.gamma
#         dz_norm = dA_norm * self.activation_derivative(self.a)
#         dvar = np.sum(dz_norm * (self.z - self.mean) * -0.5 * np.power(self.var + 1e-8, -1.5), axis=0, keepdims=True)
#         dmean = np.sum(dz_norm * -1 / np.sqrt(self.var + 1e-8), axis=0, keepdims=True)
#         dz = dz_norm * 1 / np.sqrt(self.var + 1e-8) + dvar * 2 * (self.z - self.mean) / m + dmean / m
#
#         dW = np.dot(A_prev.T, dz) / m + self.l2_reg * self.W
#         db = np.sum(dz, axis=0, keepdims=True) / m
#         dA_prev = np.dot(dz, self.W.T)
#
#         # 更新参数
#         self.W -= learning_rate * dW
#         self.b -= learning_rate * db
#         self.gamma -= learning_rate * np.sum(dA_norm * self.z_norm, axis=0, keepdims=True) / m
#         self.beta -= learning_rate * np.sum(dA_norm, axis=0, keepdims=True) / m
#
#         return dA_prev


class DenseLayer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        if self.activation != None:
            self.A = self.activation(self.Z)
        else:
            self.A = self.Z
        return self.A

    def backward(self, dA, learning_rate):
        m = self.X.shape[0]
        if self.activation_derivative != None:
            dZ = dA * self.activation_derivative(self.Z)
        else:
            dZ = dA * self.Z
        self.dW = np.dot(self.X.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return dX


class SoftmaxLayer:
    def forward(self, X):
        """
        前向传播
        :param X: 输入数据
        :return: 激活值
        """
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.a = exps / np.sum(exps, axis=1, keepdims=True)
        return self.a

    def backward(self, y_true):
        """
        反向传播
        :param y_true: 真实标签
        :return: 损失对输入的梯度
        """
        m = y_true.shape[0]
        dZ = self.a - y_true
        return dZ / m


class BaseMLP:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        self.activations = [X]
        for layer in self.layers:
            X = layer.forward(X)
            self.activations.append(X)
        return X

    def backward(self, y, learning_rate):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def train(self, X, y, epochs, learning_rate, validation_data=None):
        losses = []
        val_losses = []
        accuracies = []
        val_accuracies = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.backward(output, learning_rate)

            if epoch % 100 == 0:
                accuracy = self.accuracy(X, y)
                losses.append(loss)
                accuracies.append(accuracy)
                if validation_data:
                    val_X, val_y = validation_data
                    val_loss = self.compute_loss(val_y, self.forward(val_X))
                    val_accuracy = self.accuracy(val_X, val_y)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    print(
                        f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
                else:
                    print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')
        self.plot_training_history(losses, accuracies, val_losses, val_accuracies)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def accuracy(self, X, y):
        y_predicted = self.forward(X)
        predictions = np.argmax(y_predicted, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

    def plot_training_history(self, losses, accuracies, val_losses=None, val_accuracies=None):
        epochs = range(0, len(losses) * 100, 100)
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(epochs, losses, color='tab:red', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        if val_losses:
            ax1.plot(epochs, val_losses, color='tab:orange', label='Val Loss')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(epochs, accuracies, color='tab:blue', label='Train Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        if val_accuracies:
            ax2.plot(epochs, val_accuracies, color='tab:cyan', label='Val Accuracy')

        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.title('Training and Validation Loss and Accuracy')
        plt.show()

    def save_weights(self, file_path):
        weights = [layer.W for layer in self.layers]
        biases = [layer.b for layer in self.layers]
        with open(file_path, 'wb') as file:
            pickle.dump({'weights': weights, 'biases': biases}, file)

    def load_weights(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            weights = data['weights']
            biases = data['biases']
            for i, layer in enumerate(self.layers):
                layer.W = weights[i]
                layer.b = biases[i]


class MLPClassifier(BaseMLP):
    def __init__(self, layers, output_layer):
        super().__init__(layers)
        self.output_layer = output_layer

    def forward(self, X):
        X = super().forward(X)
        X = self.output_layer.forward(X)

        return X

    def backward(self, y, learning_rate):
        # 计算输出层的输入梯度
        dZ = self.output_layer.backward(y)
        dA = dZ

        # 反向传播隐藏层
        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backward(dA, learning_rate)

    def compute_loss(self, y, output):
        return cross_entropy_loss(y, output)

    def accuracy(self, X, y):
        y_predicted = self.forward(X)
        predictions = np.argmax(y_predicted, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)


class MLPRegressor(BaseMLP):

    def backward(self, y, learning_rate):
        dA = mean_squared_error_derivative(y, self.activations[-1])
        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backward(dA, learning_rate)

    def compute_loss(self, y, output):
        return mean_squared_error(y, output)

    def predirct(self, X):

        Y_hat = self.forward(X)
        return Y_hat

    def r2_score(self, y, output):
        # Calculate the total sum of squares (proportional to the variance of the data)
        total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)

        # Calculate the residual sum of squares
        residual_sum_of_squares = np.sum((y - output) ** 2)

        # Handle edge case where total_sum_of_squares is zero
        if total_sum_of_squares == 0:
            raise ValueError("The total sum of squares is zero, which may indicate that all true values are identical.")

        # Calculate R^2 score
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)

        return r2

    def train(self, X, y, epochs, learning_rate, validation_data=None):
        losses = []
        val_losses = []
        accuracies = []
        val_accuracies = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.backward(output, learning_rate)

            if epoch % 100 == 0:
                r2_score = self.r2_score(y, output)
                losses.append(loss)
                accuracies.append(r2_score)
                if validation_data:
                    val_X, val_y = validation_data
                    val_loss = self.compute_loss(val_y, self.forward(val_X))
                    r2_score = self.r2_score(y, output)
                    val_losses.append(val_loss)
                    val_accuracies.append(r2_score)
                    print(
                        f'Epoch {epoch}, Loss: {loss},  Val Loss: {val_loss}, r2_score: {r2_score}')
                else:
                    print(f'Epoch {epoch}, Loss: {loss}, r2_score: {r2_score}')
        self.plot_training_history(losses, accuracies, val_losses, val_accuracies)
