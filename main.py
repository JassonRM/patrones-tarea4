from deep_learning import DeepLearning
from svm import SVM

if __name__ == '__main__':
    model = DeepLearning()
    model.train()

    model2 = SVM()
    model2.train()
    model2.print()
