from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import sys
import os
from plots import *
from deep_learning import DeepLearning
from plots import plot_deep_learning, plot_svm, best_dl_model
from gui import Window

if __name__ == '__main__':
    # SVM

    # plot_svm(['degree'])
    plot_svm(['degree'])
    # model2 = SVM(x_train, x_test, y_train, y_test)
    # model2.train()
    # model2.print()

    # Deep learning
    plot_deep_learning(["epochs", "layers", "neurons", "training_set"])

    # model = best_dl_model()
    #
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"s
    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # app = QApplication(sys.argv)
    # window = Window(model)
    # sys.exit(app.exec_())
