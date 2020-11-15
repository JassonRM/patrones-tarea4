import matplotlib.pyplot as plt
from deep_learning import DeepLearning


def plot_deep_learning(plots):
    if "epochs" in plots:
        # Number of epochs
        precision = []
        recall = []
        layers = []
        for i in range(2, 21):
            model = DeepLearning(epochs=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("epochs")
        plt.show()

    if "neurons" in plots:
        # Number of layers
        precision = []
        recall = []
        layers = []
        for i in range(5, 55, 5):
            model = DeepLearning(neurons=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("neurons")
        plt.show()

    if "layers" in plots:
        # Number of layers
        precision = []
        recall = []
        layers = []
        for i in range(2, 11):
            model = DeepLearning(layers=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("layers")
        plt.show()

    if "training_set" in plots:
        # Number of layers
        precision = []
        recall = []
        layers = []
        for i in range(1000, 50000, 5000):
            model = DeepLearning(train_size=i)
            model.create_data()
            results = model.train()
            precision.append(results[0])
            recall.append(results[1])
            layers.append(i)
        plt.plot(layers, precision, label="Precision")
        plt.plot(layers, recall, label="Recall")
        plt.legend()
        plt.xlabel("training set size")
        plt.show()
