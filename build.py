def solution():
    import matplotlib.pyplot as plt
    from sklearn import datasets, svm, metrics
    from sklearn.svm import SVC
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    classifier = SVC(kernel='poly',degree=2)
    classifier.fit(data[:int(n_samples//1.2)], digits.target[:int(n_samples//1.2)])
    print(n_samples)
    print((n_samples//3))
    print(int(n_samples//1.5))
    expected = digits.target[n_samples//3:]
    predicted = classifier.predict(data[n_samples//3:])
    print("Accuracy: %s\n" % (metrics.accuracy_score(expected, predicted)))

    return classifier


solution()
