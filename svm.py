# Recognizing hand-written digits
#
# Based on: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
#
# Author: Gael Varoquaux
# License: BSD 3 clause
#
# An example showing how the scikit-learn can be used to recognize images of
# hand-written digits.


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


class SVM:

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, class_weight=None, max_iter=-1):

        self.classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                                  class_weight=class_weight, max_iter=max_iter)
        self.confusion_matrix = None

    def train(self):
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

        self.classifier.fit(x_train, y_train)

        predictions = self.predict(x_test)
        self.confusion_matrix = metrics.confusion_matrix(y_test, predictions)

    def predict(self, x):
        return self.classifier.predict(x)

    def print(self):
        print("Confusion matrix:\n%s" % self.confusion_matrix)
