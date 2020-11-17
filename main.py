from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import sys
import os
from plots import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from deep_learning import DeepLearning
from gui import Window

if __name__ == '__main__':
    plot_deep_learning(["epochs", "layers", "neurons", "training_set"])
    plot_svm(['degree'])

    # model = best_dl_model()
    # model.create_data()
    #
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # app = QApplication(sys.argv)
    # window = Window(model)
    # sys.exit(app.exec_())

    plot_svm(['degree'])
    # model2 = SVM(x_train, x_test, y_train, y_test)
    # model2.train()
    # model2.print()
