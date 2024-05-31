def load_mnist_data(train_path):
    # 加载CSV文件
    train_data = pd.read_csv(train_path).values

    # 打印数据集基本信息
    print(f"Number of rows: {train_data.shape[0]}, Number of columns: {train_data.shape[1]}")

    # 提取特征和标签
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    # 打印特征和标签的维度
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    # 打印部分数据以检查
    print("First 5 rows of X_train:")
    print(X_train[:5])
    print("First 5 labels of y_train:")
    print(y_train[:5])

    # 归一化像素值到 [0, 1]
    X_train = X_train / 255.0

    # 将标签转换为独热编码
    y_train = np.eye(10)[y_train.astype(int)]

    # 打印转换后的独热编码标签
    print("First 5 rows of one-hot encoded y_train:")
    print(y_train[:5])

    return X_train, y_train


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_index = np.random.permutation(len(data))  # 随机生成指定长度范围内不重复随机数序列
    test_set_size = int(len(data) * test_ratio)
    test_index = shuffled_index[:test_set_size]
    train_index = shuffled_index[test_set_size:]
    return data.iloc[train_index], data.ilo


# 数据的预处理

import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_california_data(datafile="./data/housing.csv"):
    """
    加载并预处理加州房价数据集。

    参数:
    datafile (str): 包含数据集的CSV文件的路径。

    返回:
    tuple: 标准化后的特征矩阵X和目标向量y。
    """
    try:
        data = pd.read_csv(datafile, sep=',')

        # Drop irrelevant columns
        data = data.drop(["longitude", "ocean_proximity"], axis=1)

        # Fill missing values in the 'total_bedrooms' column with the median value
        data["total_bedrooms"].fillna(data["total_bedrooms"].median(), inplace=True)

        # Separate features (X) and target (y)
        X = data.drop(["median_house_value"], axis=1).values
        y = data["median_house_value"].values

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y.reshape(-1, 1)

    except Exception as e:
        print(f"处理文件 {datafile} 时发生错误: {e}")


# 使用示例
if __name__ == "__main__":
    datafile = "./data/housing.csv"
    X, y = load_california_data(datafile)
    if X is not None and y is not None:
        print(f"加载的数据包含 {X.shape[0]} 条记录，每条记录有 {X.shape[1]} 个特征。")
        print(f"加载的数据包含 {y.shape[0]} 条记录，每条记录有 个特征。")
