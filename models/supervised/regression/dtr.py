from sklearn.tree import DecisionTreeRegressor
import pandas as pd


def train(X_train, y_train, X_test, y_test, params):
    clf = DecisionTreeRegressor(max_depth=int(params.get('max_depth', 5)),
                                 min_samples_split=int(params.get('min_samples_split', 2)),
                                 min_samples_leaf=int(params.get('min_samples_leaf', 3)),
                                 max_features=str(params.get('max_features', 'sqrt')),
                                 random_state=42)
    clf.fit(X_train, y_train)

    # feature importances
    try:
        importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    except Exception:
        importances = None

    return clf, {'feature_importances': importances}