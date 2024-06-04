from MLP import DenseLayer, MLPRegressor
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from MLP import DenseLayer
from MLP import relu, relu_derivative
# from MLP import MLPRegressor

# 加载加州房价数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#
# # 定义 MLPRegressor 模型
# mlp = MLPRegressor(hidden_layer_sizes=(64, 32),
#                    activation='relu',
#                    solver='adam',
#                    max_iter=1000,
#                    random_state=42)
#
# # 训练模型
# mlp.fit(X_train_scaled, y_train)
#
# # 预测目标数据
# y_train_pred = mlp.predict(X_train_scaled)
# y_test_pred = mlp.predict(X_test_scaled)
#
# # 计算训练和测试误差
# train_mse = np.mean((y_train_pred - y_train) ** 2)
# test_mse = np.mean((y_test_pred - y_test) ** 2)
#
# print(f"训练集均方误差: {train_mse:.2f}")
# print(f"测试集均方误差: {test_mse:.2f}")
#
# # 打印前5条真实房价和预测房价
# print("\n训练集前5条真实房价和预测房价:")
# for i in range(5):
#     print(f"真实房价: {y_train[i]:.2f}, 预测房价: {y_train_pred[i]:.2f}")
#
# print("\n测试集前5条真实房价和预测房价:")
# for i in range(5):
#     print(f"真实房价: {y_test[i]:.2f}, 预测房价: {y_test_pred[i]:.2f}")
# # # 数据集分为训练集和测试集
# # X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_train_normalized, test_size=0.2, random_state=42)

if __name__ == "__main__":
    layers = [
        DenseLayer(input_size=X_train_scaled.shape[1], output_size=32, activation=relu,
                   activation_derivative=relu_derivative),
        DenseLayer(input_size=32, output_size=1, activation=None,
                   activation_derivative=None),

        # Linear activation
    ]

    # 实例化MLP回归器
    mlp_regressor = MLPRegressor(layers)

    # 训练模型
    mlp_regressor.train(X_train_scaled, y_train, epochs=1000, learning_rate=0.01)

    predicted_values = mlp_regressor.predict(X_test)

    # 输出结果
    print("Predicted values shape:", predicted_values.shape)
    print("Sample predicted values:", predicted_values[:5])  # 打印前5个预测值
    print("Sample target values:", y_test[:5])  # 打印前5个真实目标值