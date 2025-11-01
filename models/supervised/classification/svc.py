from sklearn.svm import SVC


def train(X_train, y_train, X_test, y_test, params):
    clf = SVC(C=float(params.get('C', 1.0)), kernel=params.get('kernel', 'rbf'), probability=True, random_state=42)
    clf.fit(X_train, y_train)
    return clf, {}