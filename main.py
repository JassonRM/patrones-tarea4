from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import sys
import os
from plots import plot_deep_learning, plot_svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from deep_learning import DeepLearning
from gui import Window




if __name__ == '__main__':
    train_size = 50000
    val_size = 10000

    # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape matrices to 784-length vectors for training
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Save images for validation
    if train_size + val_size <= 60000:
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
    else:
        print("The MNIST data set only has 60000 images, train_size and val_size can't be more than 60000")
        exit(-1)

    plot_deep_learning(x_train, y_train, x_val, y_val, x_test, y_test, ["epochs", "layers", "neurons", "training_set"])

    # model = best_dl_model()
    # model.create_data()
    #
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # app = QApplication(sys.argv)
    # window = Window(model)
    # sys.exit(app.exec_())

    plot_svm('kernel')
    # model2 = SVM(x_train, x_test, y_train, y_test)
    # model2.train()
    # model2.print()
