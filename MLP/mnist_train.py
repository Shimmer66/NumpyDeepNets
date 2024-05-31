# 定义 MLP 分类模型
from MLP import DenseLayer, SoftmaxLayer, MLPClassifier, relu_derivative, relu
from load_data import load_mnist_data

if __name__ == '__main__':
    layers = [
        DenseLayer(input_size=784, output_size=512, activation=relu, activation_derivative=relu_derivative),
        DenseLayer(input_size=512, output_size=256, activation=relu, activation_derivative=relu_derivative),
        DenseLayer(input_size=256, output_size=10, activation=relu, activation_derivative=relu_derivative)
    ]

    output_layer = SoftmaxLayer()
    mlp_classifier = MLPClassifier(layers, output_layer)
    X_train, y_train = load_mnist_data('./data/mnist_train.csv')
    # 训练模型
    mlp_classifier.train(X_train, y_train, epochs=1000, learning_rate=0.5)

    # 评估模型
    train_accuracy = mlp_classifier.accuracy(X_train, y_train)

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    # 保存和加载权重
    mlp_classifier.save_weights('./weights/mlp_classifier_weights.pkl')
    # mlp_classifier.load_weights('mlp_classifier_weights.pkl')
