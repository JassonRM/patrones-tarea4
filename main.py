import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from plots import *
from gui import Window

if __name__ == '__main__':
    mode = 1
    if mode == 1:
        model = best_dl_model(retrain=False)
    else:
        model = best_svm_model(retrain=False)

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = Window(model)
    sys.exit(app.exec_())

    # plot_deep_learning(["epochs", "layers", "neurons", "training_set"])
    # plot_svm(['kernel', 'C'])
