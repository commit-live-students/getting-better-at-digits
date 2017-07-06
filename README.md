# Getting better at digits

### The Setting
Let us put together the code we had used in the class for image classification using digits.

    import matplotlib.pyplot as plt
    from sklearn import datasets, svm, metrics
    from sklearn.linear_model import LogisticRegression

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier
    classifier = LogisticRegression()

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    print("Accuracy: %s\n" % (metrics.accuracy_score(expected, predicted)))
    
The accuracy using Logistic Regression is 91.7%.

### Problem Statement
Write a function called `digits()` which 

* accepts the parameters `X_train_digits` and `y_train_digits` datasets (available in your environment) for training, and 
* returns the trained model as output

### Hint
* We'll create `X_train_digits`, `y_train_digits`, `X_test` and `y_test`
* `X_train_digits` and `y_train_digits` will be available in student's environment
* `X_test` and `y_test` will be available to us to evaluate the output from student's function
