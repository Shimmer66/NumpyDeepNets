from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from MLP import DenseLayer, relu_derivative, relu, MLPRegressor

# 获取California住房数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    layers = [
        DenseLayer(input_size=X_train.shape[1], output_size=32, activation=lambda x: 1,
                   activation_derivative=lambda x: 1),
        DenseLayer(input_size=32, output_size=1, activation=lambda x: 1,
                   activation_derivative=lambda x: 1),

        # Linear activation
    ]

    # 实例化MLP回归器
    mlp_regressor = MLPRegressor(layers)

    # 训练模型
    mlp_regressor.train(X_train, y_train, epochs=1000, learning_rate=0.01)
