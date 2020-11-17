# Recognizing hand-written digits
#
# Based on: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
#
# Author: Gael Varoquaux
# License: BSD 3 clause
#
# An example showing how the scikit-learn can be used to recognize images of
# hand-written digits.


from sklearn import svm, metrics
from sklearn.metrics import precision_recall_fscore_support


class SVM:

    def __init__(self, x_train, x_test, y_train, y_test, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 mode='macro'):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.confusion_matrix = None
        self.precision = None
        self.recall = None
        self.avg_precision = 0
        self.avg_recall = 0
        self.classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        if mode != 'weighted':
            mode = 'macro'
        self.mode = mode

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.predict(self.x_test)
        self.confusion_matrix = metrics.confusion_matrix(self.y_test, predictions)
        score = precision_recall_fscore_support(self.y_test, predictions)
        self.precision = score[0]
        self.recall = score[1]
        avg = precision_recall_fscore_support(self.y_test, predictions, average=self.mode)
        self.avg_precision = avg[0]
        self.avg_recall = avg[1]
        return self.avg_precision, self.avg_recall

    def predict(self, x):
        return self.classifier.predict(x)

    def print(self):
        print("Confusion matrix:\n%s" % self.confusion_matrix)
        print("Precision:\n%s" % self.precision)
        print("Recall:\n%s" % self.recall)
        print(self.avg_recall)
        print(self.avg_precision)
        print(self.classifier.kernel)
        print(self.classifier.gamma)
        print(self.classifier.C)
        print(self.classifier.degree)
