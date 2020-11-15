from plots import plot_deep_learning
from svm import SVM

if __name__ == '__main__':
    plot_deep_learning(["epochs", "layers", "neurons", "training_set"])

    # model2 = SVM()
    # model2.train()
    # model2.print()
