import matplotlib.pyplot as plt
from deep_learning import DeepLearning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from svm import SVM


def plot_deep_learning(plots):
    if "epochs" in plots:
        # Number of epochs
        precision = []
        recall = []
        layers = []
        for i in range(2, 21):
            model = DeepLearning(epochs=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("epochs")
        plt.show()

    if "neurons" in plots:
        # Number of layers
        precision = []
        recall = []
        layers = []
        for i in range(5, 55, 5):
            model = DeepLearning(neurons=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("neurons")
        plt.show()

    if "layers" in plots:
        # Number of layers
        precision = []
        recall = []
        layers = []
        for i in range(2, 11):
            model = DeepLearning(layers=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("layers")
        plt.show()

    if "training_set" in plots:
        # Number of layers
        precision = []
        recall = []
        layers = []
        for i in range(1000, 50000, 5000):
            model = DeepLearning(train_size=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("training set size")
        plt.show()


def plot_svm(plots):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    if 'kernel' in plots:
        precision = []
        recall = []
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernels:
            svm = SVM(x_train, x_test, y_train, y_test, kernel=kernel)
            result = svm.train()
            precision.append(result[0])
            recall.append(result[1])
        plt.plot(precision, recall, 'bo')
        for xyk in zip(precision, recall, kernels):
            plt.annotate(xyk[2], xyk[0:2])
        # plt.xlim((0.4, 1))
        # plt.ylim((0.4, 1))
        plt.show()
