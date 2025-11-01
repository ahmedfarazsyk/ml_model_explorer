import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def build_transformer(df):
    transformer = ColumnTransformer(transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), make_column_selector(dtype_include=object)),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), make_column_selector(dtype_include=[int, float]))
    ], remainder='drop')
    transformer.set_output(transform='pandas')
    return transformer


def preprocess_features(transformer, X):
    return transformer.fit_transform(X)


def encode_target(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le

def scale_target(y):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    y = np.array(y).reshape(-1, 1)
    y_imputed = imputer.fit_transform(y)
    y_sca = scaler.fit_transform(y_imputed).ravel()
    return y_sca


def only_numeric_check(X):
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return non_numeric