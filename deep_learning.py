import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical # One hot encoding
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

model = Sequential()

def deep_learning():
	# The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Reshape matrices to 784-length vectors for training
	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_val = x_train[-10000:] # Save 10000 images for validation
	y_val = y_train[-10000:]
	x_train = x_train[:-10000]
	y_train = y_train[:-10000]

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

	# Configure the model
	model.add(Dense(512, activation='relu', input_shape=(784,)))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	# Summarize the built model
	model.summary()

	# Let's use the Adam optimizer for learning
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	## TensorBoard Callback
	tcb = TensorBoard()

	# Train the model
	history_callback = model.fit(x_train, y_train,
								 batch_size=128, epochs=5,
								 verbose=1,
								 validation_data=(x_val, y_val),
								 callbacks=[tcb])

	score = model.evaluate(x_test, y_test)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


	# Plot accuracy and loss
	acc = history_callback.history['accuracy']
	loss = history_callback.history['loss']

	val_acc = history_callback.history['val_accuracy']
	val_loss = history_callback.history['val_loss']

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
	ground_truth = y_test.argmax(axis=1)
	predictions = predict(x_test).argmax(axis=1)

	confusion = confusion_matrix(ground_truth, predictions)
	print(confusion)

	report = classification_report(ground_truth, predictions)
	print(report)

	stats = precision_recall_fscore_support(ground_truth, predictions, average="weighted")
	print("Precision={0}, Recall={1}".format(stats[0], stats[1]))

def predict(x):
	return model.predict(x)