import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from plots import plot_deep_learning, plot_svm
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
