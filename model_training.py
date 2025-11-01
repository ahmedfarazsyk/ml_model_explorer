import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# import model wrappers
from models.supervised.classification.dtc import train as train_dtc
from models.supervised.classification.rfc import train as train_rfc
from models.supervised.classification.svc import train as train_svc

from models.supervised.regression.dtr import train as train_dtr
from models.supervised.regression.rfr import train as train_rfr
from models.supervised.regression.svr import train as train_svr

from models.unsupervised.kmeans_mod import run as run_kmeans
from models.unsupervised.agglomerative_mod import run as run_agg
from models.unsupervised.dbscan_mod import run as run_dbscan


def train_classifiers(X, y, model_name, params, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if model_name == 'Decision Tree':
        model, extra = train_dtc(X_train, y_train, X_test, y_test, params)
    elif model_name == 'Random Forest':
        model, extra = train_rfc(X_train, y_train, X_test, y_test, params)
    else:
        model, extra = train_svc(X_train, y_train, X_test, y_test, params)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    clf_report = classification_report(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,4))
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.tight_layout()

    result = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'classification_report': clf_report,
        'confusion_fig': fig,
        'feature_importances': extra.get('feature_importances') if extra else None
    }
    return result


def train_regressors(X, y, model_name, params, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if model_name == 'Decision Tree':
        model, extra = train_dtr(X_train, y_train, X_test, y_test, params)
    elif model_name == 'Random Forest':
        model, extra = train_rfr(X_train, y_train, X_test, y_test, params)
    else:
        model, extra = train_svr(X_train, y_train, X_test, y_test, params)

    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, preds, alpha=0.6, color='steelblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name} — Predicted vs Actual')
    plt.tight_layout()

    # ----- Visualization: Residual Plot -----
    residuals = y_test - preds
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(preds, residuals, alpha=0.6, color='darkorange')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{model_name} — Residual Plot')
    plt.tight_layout()

    result = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'fig':fig,
        'fig2':fig2,
        'feature_importances': extra.get('feature_importances') if extra else None
    }
    return result


def run_unsupervised(X, algo_name, params):
    if algo_name == 'KMeans':
        labels, sil, fig = run_kmeans(X, params)
    elif algo_name == 'Agglomerative':
        labels, sil, fig = run_agg(X, params)
    else:
        labels, sil, fig = run_dbscan(X, params)

    return {'labels': labels, 'silhouette': sil, 'cluster_fig': fig}