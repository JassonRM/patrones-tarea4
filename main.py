from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import sys
import os
from plots import *

from gui import Window

if __name__ == '__main__':
    mode = 0
    if mode == 1:
        model = best_dl_model()
    else:
        model = best_svm_model()

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = Window(model)
    sys.exit(app.exec_())

    # x_train, y_train, x_val, y_val, x_test, y_test = create_data()
    # svm = SVM(x_train, x_test, y_train, y_test, C=1, kernel='poly', degree=2, gamma=0.01)
    # svm.train()
    # svm.save_model('best_svm_model')
    # svm.print()

    # plot_deep_learning(["epochs", "layers", "neurons", "training_set"])
    # plot_svm(['kernel', 'C'])
