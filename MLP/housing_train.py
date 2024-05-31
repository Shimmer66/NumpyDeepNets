# 定义 MLP 回归模型
import pickle

from MLP import DenseLayer, relu_derivative, relu, MLPRegressor
from load_data import load_california_data

if __name__ == "__main__":
    X_train, y_train = load_california_data()
    layers = [
        DenseLayer(input_size=7, output_size=256, activation=relu, activation_derivative=relu_derivative),
        DenseLayer(input_size=256, output_size=64, activation=relu, activation_derivative=relu_derivative),
        DenseLayer(input_size=64, output_size=1, activation=lambda x: x, activation_derivative=lambda x: 1)
        # Linear activation
    ]

    # 实例化MLP回归器
    mlp_regressor = MLPRegressor(layers)

    # 训练模型
    mlp_regressor.train(X_train, y_train, epochs=1000, learning_rate=0.5)


    # 保存模型权重
    # with open('./weights/mlp_regressor_weights.pkl', 'wb') as f:
    #     pickle.dump(mlp_regressor.get_weights(), f)
