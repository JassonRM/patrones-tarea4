import matplotlib.pyplot as plt
from deep_learning import DeepLearning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from svm import SVM
import os
from create_data import create_data


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
        plt.show()
    if 'gamma' in plots:
        precision = []
        recall = []
        highest_recall = 0
        highest_precision = 0
        highest_recall_val = 0
        highest_precision_val = 0
        for gamma in range(1, 100, 1):
            g = gamma / 10000
            svm = SVM(x_train, x_test, y_train, y_test, gamma=g)
            result = svm.train()
            precision.append(result[0])
            recall.append(result[1])
            if result[1] > highest_recall:
                highest_recall = result[1]
                highest_recall_val = g
            if result[0] > highest_precision:
                highest_precision = result[0]
                highest_precision_val = g

        plt.title('Highest precision with %.4f, highest recall with %.4f' % (highest_precision_val, highest_recall_val))
        plt.plot(precision, recall, 'bo')
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.show()
    if 'C' in plots:
        precision = []
        recall = []
        highest_recall = 0
        highest_recall_val = 0
        highest_precision = 0
        highest_precision_val = 0
        for _c in range(10, 1000, 10):
            c = _c / 100
            svm = SVM(x_train, x_test, y_train, y_test, C=c)
            result = svm.train()
            precision.append(result[0])
            recall.append(result[1])
            if result[1] > highest_recall:
                highest_recall = result[1]
                highest_recall_val = c
            if result[0] > highest_precision:
                highest_precision = result[0]
                highest_precision_val = c

        plt.title('Highest precision with %.4f, highest recall with %.4f' % (highest_precision_val, highest_recall_val))
        plt.plot(precision, recall, 'bo')
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.show()
    if 'degree' in plots:
        precision = []
        recall = []
        highest_recall = 0
        highest_recall_val = 0
        highest_precision = 0
        highest_precision_val = 0
        for degree in range(1, 25, 1):
            svm = SVM(x_train, x_test, y_train, y_test, degree=degree, kernel='poly')
            result = svm.train()
            precision.append(result[0])
            recall.append(result[1])
            if result[1] > highest_recall:
                highest_recall_val = degree
                highest_recall = result[1]
            if result[0] > highest_precision:
                highest_precision = result[0]
                highest_precision_val = degree

        plt.title('Highest precision with %.4f, highest recall with %.4f' % (highest_precision_val, highest_recall_val))
        plt.plot(precision, recall, 'bo')
        plt.ylabel('Recall')
        plt.xlabel('Precision')

        plt.show()


def plot_best_svm_overall():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    precision = []
    recall = []
    highest_score = 0
    highest_model = None
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for _c in range(10, 1000, 10):
            c = _c / 100
            if kernel == 'poly' or kernel == 'sigmoid' or kernel == 'rbf':
                for gamma in range(1, 100, 1):
                    g = gamma / 10000
                    if kernel == 'poly':
                        for degree in range(1, 25, 1):
                            svc = SVM(x_train, x_test, y_train, y_test, gamma=g, C=c, kernel=kernel, degree=degree)
                            result = svc.train()
                            precision.append(result[0])
                            recall.append(result[1])
                            if (result[0] + result[1]) / 2 > highest_score:
                                highest_score = (result[0] + result[1]) / 2
                                highest_model = svc
                    else:
                        svc = SVM(x_train, x_test, y_train, y_test, gamma=g, C=c, kernel=kernel)
                        result = svc.train()
                        precision.append(result[0])
                        recall.append(result[1])
                        if (result[0] + result[1]) / 2 > highest_score:
                            highest_score = (result[0] + result[1]) / 2
                            highest_model = svc
            else:
                svc = SVM(x_train, x_test, y_train, y_test, C=c, kernel=kernel)
                result = svc.train()
                precision.append(result[0])
                recall.append(result[1])
                if (result[0] + result[1]) / 2 > highest_score:
                    highest_score = (result[0] + result[1]) / 2
                    highest_model = svc
    plt.plot(precision, recall, 'bo')
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.show()

    highest_model.print()



def plot_deep_learning(plots):
    x_train, y_train, x_val, y_val, x_test, y_test = create_data()
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

    if "training_set" in plots:
        # Training set size
        precision = []
        recall = []
        for i in range(1000, 50000, 5000):
            x_train, y_train, x_val, y_val, x_test, y_test = create_data(training_size=i)
            model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test)
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
        plt.scatter(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Training set size")
        plt.show()


def best_dl_model(retrain=False):
    if os.path.exists("best_dl_model") and not retrain:
        model = DeepLearning(None, None, None, None, None, None)
        model.load_model("best_dl_model")
    else:
        best_model = None
        best_precision = 0
        best_recall = 0
        best_training_set = None
        best_epochs = None
        best_neurons = None
        best_layers = None

        for training_set in range(100, 50000, 5000):
            x_train, y_train, x_val, y_val, x_test, y_test = create_data(training_size=training_set)
            for epochs in range(2, 21, 2):
                for neurons in range(5, 55, 5):
                    for layers in range(2, 11):
                        model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test, epochs=epochs, neurons=neurons, layers=layers)
                        precision, recall = model.train()
                        if precision + recall > best_precision + best_recall:
                            best_model = model
                            best_precision = precision
                            best_recall = recall
                            best_training_set = training_set
                            best_epochs = epochs
                            best_neurons = neurons
                            best_layers = layers

        print("Best configuration:")
        print("Precision: ", best_precision)
        print("Recall: ", best_recall)
        print("Training set size: ", best_training_set)
        print("Epochs: ", best_epochs)
        print("Neurons: ", best_neurons)
        print("Layers: ", best_layers)
        print("----------------------")
        best_model.print()
        best_model.save_model("best_dl_model")
    return best_model
