import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from deep_learning import DeepLearning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from svm import SVM
import os
from create_data import create_data


def plot_svm(plots):
    x_train, y_train, x_val, y_val, x_test, y_test = create_data()
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
        plt.suptitle('Kernel Variation', fontsize=14, fontweight='bold')
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.show()
    if 'gamma' in plots:
        precision = []
        recall = []
        highest_recall = 0
        highest_precision = 0
        highest_recall_val = 0
        highest_precision_val = 0
        for gamma in [0.001, 0.01, 0.1, 1, 10]:
            svm = SVM(x_train, x_test, y_train, y_test, gamma=gamma)
            result = svm.train()
            precision.append(result[0])
            recall.append(result[1])
            if result[1] > highest_recall:
                highest_recall = result[1]
                highest_recall_val = gamma
            if result[0] > highest_precision:
                highest_precision = result[0]
                highest_precision_val = gamma

        plt.suptitle('Gamma Variation', fontsize=14, fontweight='bold')
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
        for c in [0.1, 1, 10, 100, 1000]:
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

        plt.suptitle('C Variation', fontsize=14, fontweight='bold')
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
        for degree in range(1, 7, 1):
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

        plt.suptitle('Degree Variation', fontsize=14, fontweight='bold')
        plt.title('Highest precision with %.4f, highest recall with %.4f' % (highest_precision_val, highest_recall_val))
        plt.plot(precision, recall, 'bo')
        plt.ylabel('Recall')
        plt.xlabel('Precision')

        plt.show()


def best_svm_model(retrain=False):
    if os.path.exists("best_svm_model") and not retrain:
        best_model = SVM(None, None, None, None)
        best_model.load_model("best_svm_model")
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = create_data(training_size=500, test_size=1618)

        precision = []
        recall = []
        highest_score = 0
        best_model = None
        for kernel in ['linear', 'poly', 'rbf']:
            for c in [0.1, 1, 10, 100, 1000]:
                if kernel == 'poly' or kernel == 'rbf':
                    for gamma in [0.001, 0.01, 0.1, 1, 10]:
                        if kernel == 'poly':
                            for degree in range(1, 5, 1):
                                svc = SVM(x_train, x_test, y_train, y_test, gamma=gamma, C=c, kernel=kernel,
                                          degree=degree)
                                result = svc.train()
                                precision.append(result[0])
                                recall.append(result[1])
                                if (result[0] + result[1]) / 2 > highest_score:
                                    highest_score = (result[0] + result[1]) / 2
                                    best_model = svc
                        else:
                            svc = SVM(x_train, x_test, y_train, y_test, gamma=gamma, C=c, kernel=kernel)
                            result = svc.train()
                            precision.append(result[0])
                            recall.append(result[1])
                            if (result[0] + result[1]) / 2 > highest_score:
                                highest_score = (result[0] + result[1]) / 2
                                best_model = svc
                else:
                    svc = SVM(x_train, x_test, y_train, y_test, C=c, kernel=kernel)
                    result = svc.train()
                    precision.append(result[0])
                    recall.append(result[1])
                    if (result[0] + result[1]) / 2 > highest_score:
                        highest_score = (result[0] + result[1]) / 2
                        best_model = svc
        plt.ylabel('Recall')
        plt.xlabel('Precision')

        scores = np.vstack((np.array(precision), np.array(recall))).T
        pareto = identify_pareto(scores)
        pareto_front = scores[pareto]
        ax = plt.gca()

        for corner in pareto_front:
            rect = patches.Rectangle((0, 0), corner[0], corner[1], facecolor='gray', alpha=0.005)
            ax.add_patch(rect)
        plt.scatter(precision, recall)

        plt.show()
        best_model.save_model('best_svm_model')
        best_model.print()
    return best_model


def plot_deep_learning(plots):
    x_train, y_train, x_val, y_val, x_test, y_test = create_data()
    if "epochs" in plots:
        # Number of epochs
        precision = []
        recall = []
        for i in range(2, 21, 2):
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
        for i in range(5, 60, 10):
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
        for i in range(2, 11, 2):
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
        for i in range(10000, 60000, 10000):
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
        best_model = DeepLearning(None, None, None, None, None, None)
        best_model.load_model("best_dl_model")
    else:
        best_model = None
        best_precision = 0
        best_recall = 0
        best_training_set = None
        best_epochs = None
        best_neurons = None
        best_layers = None

        for training_set in range(10000, 60000, 10000):
            x_train, y_train, x_val, y_val, x_test, y_test = create_data()
            for epochs in range(2, 21, 2):
                for neurons in range(5, 60, 10):
                    for layers in range(2, 11, 2):
                        model = DeepLearning(x_train, y_train, x_val, y_val, x_test, y_test, epochs=epochs,
                                             neurons=neurons, layers=layers)
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


# Code taken from https://pythonhealthcare.org/tag/pareto-front/
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]
