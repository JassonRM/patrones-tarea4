from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import sys
from sklearn import datasets
import os

from sklearn.model_selection import train_test_split

from plots import plot_deep_learning
from deep_learning import DeepLearning
from svm import SVM
from gui import Window


def best_dl_model():
    model = DeepLearning(epochs=15, neurons=50, layers=6, train_size=50000, verbose=1)
    if os.path.exists("best_dl_model"):
        model.load_model("best_dl_model")
    else:
        model.create_data()
        model.train()
        model.save_model("best_dl_model")
    return model


if __name__ == '__main__':
    # plot_deep_learning(["epochs", "layers", "neurons", "training_set"])

    model = best_dl_model()
    model.create_data()

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = Window(model)
    sys.exit(app.exec_())

    # digits = datasets.load_digits()
    # n_samples = len(digits.images)
    # data = digits.images.reshape((n_samples, -1))
    # x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
    # model2 = SVM(x_train, x_test, y_train, y_test)
    # model2.train()
    # model2.print()
