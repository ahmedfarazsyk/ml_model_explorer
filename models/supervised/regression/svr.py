from sklearn.svm import SVR


def train(X_train, y_train, X_test, y_test, params):
    clf = SVR(C=float(params.get('C', 1.0)), kernel=params.get('kernel', 'rbf'))
    clf.fit(X_train, y_train)
    return clf, {}