def solution():
    '''
    Enter your code here
    '''
    # Load dataset
    from sklearn import datasets
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Import RandomForest classifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    # Train test Split
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.7, random_state=42)

    # Fit on the training data
    classifier.fit(X_train, y_train)

    return classifier
