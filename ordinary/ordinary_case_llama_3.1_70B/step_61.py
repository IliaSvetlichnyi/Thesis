import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def step_61(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if y_train_pred.dtype.kind in 'bi':
        # classification metrics
        labels = np.unique(y_train)
        if len(labels) > 2:
            accuracy_train = accuracy_score(y_train, y_train_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            precision_train = precision_score(y_train, y_train_pred, average='weighted')
            precision_test = precision_score(y_test, y_test_pred, average='weighted')
            recall_train = recall_score(y_train, y_train_pred, average='weighted')
            recall_test = recall_score(y_test, y_test_pred, average='weighted')
            f1_train = f1_score(y_train, y_train_pred, average='weighted')
            f1_test = f1_score(y_test, y_test_pred, average='weighted')
        else:
            accuracy_train = accuracy_score(y_train, y_train_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            precision_train = precision_score(y_train, y_train_pred, average='binary')
            precision_test = precision_score(y_test, y_test_pred, average='binary')
            recall_train = recall_score(y_train, y_train_pred, average='binary')
            recall_test = recall_score(y_test, y_test_pred, average='binary')
            f1_train = f1_score(y_train, y_train_pred, average='binary')
            f1_test = f1_score(y_test, y_test_pred, average='binary')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        evaluation_results = {
            'train': [accuracy_train, precision_train, recall_train, f1_train],
            'test': [accuracy_test, precision_test, recall_test, f1_test]
        }
    else:
        # regression metrics
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        metrics = ['r2_score', 'mse', 'rmse', 'mae']
        evaluation_results = {
            'train': [r2_train, mse_train, rmse_train, mae_train],
            'test': [r2_test, mse_test, rmse_test, mae_test]
        }

    return evaluation_results, metrics