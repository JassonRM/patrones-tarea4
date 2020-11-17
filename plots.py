import matplotlib.pyplot as plt
from deep_learning import DeepLearning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from svm import SVM
import os

epochs = 0
neurons = 0
layers = 0
training_set = 0


def plot_deep_learning(x_train, y_train, x_val, y_val, x_test, y_test, plots):
    if "epochs" in plots:
        # Number of epochs
        precision = []
        recall = []
        for i in range(2, 21):
            model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test, epochs=i)
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
        plt.scatter(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Epochs")
        plt.show()

    if "neurons" in plots:
        # Number of layers
        precision = []
        recall = []
        for i in range(5, 55, 5):
            model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test, neurons=i)
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
        plt.scatter(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Neurons")
        plt.show()

    if "layers" in plots:
        # Number of layers
        precision = []
        recall = []
        for i in range(2, 11):
            model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test, layers=i)
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
        plt.scatter(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Layers")
        plt.show()

# if "training_set" in plots:
#     # Training set size
#     precision = []
#     recall = []
#     for i in range(1000, 50000, 5000):
#         model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test, train_size=i)
#         results = model.train()
#         precision.append(results[0])
#         recall.append(results[1])
# plt.scatter(recall, precision)
# plt.ylabel("Precision")
# plt.xlabel("Recall")
#     plt.title("Training set size")
#     plt.show()


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


def best_dl_model():
    model = DeepLearning(epochs=epochs, neurons=neurons, layers=layers, train_size=training_set, verbose=1)
    if os.path.exists("best_dl_model"):
        model.load_model("best_dl_model")
    else:
        model.create_data()
        model.train()
        model.save_model("best_dl_model")
    return model
