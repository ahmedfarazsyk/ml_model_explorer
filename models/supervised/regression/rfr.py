from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def train(X_train, y_train, X_test, y_test, params):
    clf = RandomForestRegressor(n_estimators=int(params.get('n_estimators', 100)),
                                 max_depth=None if params.get('max_depth') in (None, 'None') else int(params.get('max_depth', 10)),
                                 min_samples_leaf=int(params.get('min_samples_leaf', 3)),
                                 max_features=str(params.get('max_features', 'sqrt')),
                                 random_state=42,
                                 n_jobs=-1)
    clf.fit(X_train, y_train)
    try:
        importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    except Exception:
        importances = None
    return clf, {'feature_importances': importances}