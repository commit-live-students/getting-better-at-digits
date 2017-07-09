import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression


def solution():
    '''
    Enter your code here
    '''
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    classifier = svm.SVC(gamma =0.001)
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])
    return classifier
