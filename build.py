import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets,svm, metrics

def solution():
    digits = datasets.load_digits()
    # print digits
    n_samples = len(digits.images)
    # n_samples
    data = digits.images.reshape((n_samples, -1))

    #Using SVC Algorithm to classify
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # expected = digits.target[n_samples //2:]
    # predicted = classifier.predict(data[n_samples // 2:])

    # print metrics.accuracy_score(expected, predicted)
    return classifier
