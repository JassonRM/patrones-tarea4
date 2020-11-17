from tensorflow.keras.datasets import mnist


def create_data(training_size=50000, validation_size=10000, test_size=10000):

    # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape matrices to 784-length vectors for training
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Adjust sizes
    if training_size + validation_size <= 60000:
        x_val = x_train[-validation_size:]
        y_val = y_train[-validation_size:]
        x_train = x_train[:training_size]
        y_train = y_train[:training_size]
        x_test = x_test[:test_size]
        y_test = y_test[:test_size]
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        print("The MNIST data set only has 60000 images, train_size and val_size can't be more than 60000")
        return