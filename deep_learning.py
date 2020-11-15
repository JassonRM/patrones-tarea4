import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical  # One hot encoding
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt


class DeepLearning:
    def __init__(self, epochs=5, layers=2, neurons=518, train_size=50000, val_size=10000, verbose=0):
        self.epochs = epochs
        self.layers = layers
        self.neurons = neurons
        self.train_size = train_size
        self.val_size = val_size
        self.verbose = verbose
        self.model = Sequential()
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.history_callback = None
        self.confusion_matrix = None
        self.report = None
        self.precision = None
        self.recall = None

    def load_model(self, path):
        self.model = load_model(path)

    def save_model(self, path):
        self.model.save(path)

    def create_data(self):
        # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape matrices to 784-length vectors for training
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        # Save images for validation
        if (self.train_size + self.val_size <= 60000):
            x_val = x_train[-self.val_size:]
            y_val = y_train[-self.val_size:]
            x_train = x_train[:self.train_size]
            y_train = y_train[:self.train_size]
        else:
            print("The MNIST data set only has 60000 images, train_size and val_size can't be more than 60000")
            return

        # Normalize between 0 and 1
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_val /= 255
        x_test /= 255

        # One hot encoding of the output class
        classes = 10
        y_train = to_categorical(y_train, classes)
        y_val = to_categorical(y_val, classes)
        y_test = to_categorical(y_test, classes)

        # Save data
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        # Configure the model
        self.model.add(Dense(self.neurons, activation='relu', input_shape=(784,)))
        for i in range(self.layers - 2):
            self.model.add(Dense(self.neurons, activation='relu'))
        self.model.add(Dense(self.neurons, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        # Summarize the built model
        if self.verbose == 1:
            self.model.summary()

        # Let's use the Adam optimizer for learning
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.history_callback = self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=self.epochs, verbose=self.verbose,
                                               validation_data=(self.x_val, self.y_val))

        return self.evaluate()

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # Calculate confusion matrix and report
        ground_truth = self.y_test.argmax(axis=1)
        predictions = self.predict(self.x_test).argmax(axis=1)
        self.confusion_matrix = confusion_matrix(ground_truth, predictions)
        self.report = classification_report(ground_truth, predictions)
        stats = precision_recall_fscore_support(ground_truth, predictions, average="weighted")
        self.precision = stats[0]
        self.recall = stats[1]

        return self.precision, self.recall

    def predict(self, x):
        return self.model.predict(x)

    def print(self):
        # Plot accuracy and loss
        acc = self.history_callback.history['accuracy']
        loss = self.history_callback.history['loss']
        val_acc = self.history_callback.history['val_accuracy']
        val_loss = self.history_callback.history['val_loss']

        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
        ax1.plot(acc, label="Training Accuracy")
        ax1.plot(val_acc, label="Validation Accuracy")
        ax2.plot(loss, label="Training Loss")
        ax2.plot(val_loss, label="Validation Loss")

        ax2.set_xlabel('epochs')
        ax1.legend()
        ax2.legend()
        plt.show()

        # Show confusion matrix and statistics
        print(self.confusion_matrix)
        print(self.report)
        print("Precision={0}, Recall={1}".format(self.precision, self.recall))
