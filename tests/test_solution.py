from unittest import TestCase


class TestSolution(TestCase):
    def test_solution(self):
        from build import solution
        from sklearn import datasets, svm, metrics

        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        clf = solution()
        expected = digits.target[n_samples // 2:]
        predicted = clf.predict(data[n_samples // 2:])

        self.assertGreaterEqual(metrics.accuracy_score(expected, predicted), 0.96)
