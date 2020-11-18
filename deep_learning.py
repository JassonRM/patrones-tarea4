import os

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical  # One hot encoding
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# CUDA Setup
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# This class is based on the tutorial presented in
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# as well as lesson 17 from Pablo Alvarado Moya's Introduction to pattern recognition course at TEC.

class DeepLearning:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, epochs=5, layers=2, neurons=512, verbose=0):
        self.epochs = epochs
        self.layers = layers
        self.neurons = neurons
        self.verbose = verbose
        self.model = Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.history_callback = None
        self.confusion_matrix = None
        self.report = None
        self.precision = None
        self.recall = None

    def load_model(self, path):
        self.model = load_model(path)

    def save_model(self, path):
        self.model.save(path)

    def train(self):
        # Normalize between 0 and 1
        self.x_train = self.x_train.astype('float32')
        self.x_val = self.x_val.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_val /= 255
        self.x_test /= 255

        # One hot encoding of the output class
        classes = 10
        self.y_train = to_categorical(self.y_train, classes)
        self.y_val = to_categorical(self.y_val, classes)
        self.y_test = to_categorical(self.y_test, classes)

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
        self.history_callback = self.model.fit(self.x_train, self.y_train, batch_size=100, epochs=self.epochs,
                                               verbose=self.verbose,
                                               validation_data=(self.x_val, self.y_val))

        # Evaluate
        score = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # Calculate confusion matrix and report
        ground_truth = self.y_test.argmax(axis=1)
        predictions = self.model.predict(self.x_test).argmax(axis=1)
        self.confusion_matrix = confusion_matrix(ground_truth, predictions)
        self.report = classification_report(ground_truth, predictions)
        stats = precision_recall_fscore_support(ground_truth, predictions, average="weighted")
        self.precision = stats[0]
        self.recall = stats[1]

        return self.precision, self.recall

    def predict(self, x):
        x = x.astype('float32')
        x /= 255
        result = self.model.predict(x)[0]
        max_value = 0
        for i in range(10):
            if result[i] > max_value:
                max_value = result[i]
                index = i
        return index

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
