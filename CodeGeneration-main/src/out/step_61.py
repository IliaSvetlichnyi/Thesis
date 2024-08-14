
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def step_61(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if y_train.dtype.kind in ['i', 'f']:  # regression
        metrics = ['R^2', 'MSE', 'RMSE']
        evaluation_results = {
            'Train': [r2_score(y_train, y_train_pred), mean_squared_error(y_train, y_train_pred), mean_squared_error(y_train, y_train_pred, squared=False)],
            'Test': [r2_score(y_test, y_test_pred), mean_squared_error(y_test, y_test_pred), mean_squared_error(y_test, y_test_pred, squared=False)]
        }
    else:  # classification
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        evaluation_results = {
            'Train': [accuracy_score(y_train, y_train_pred), precision_score(y_train, y_train_pred, average='weighted'), recall_score(y_train, y_train_pred, average='weighted'), f1_score(y_train, y_train_pred, average='weighted')],
            'Test': [accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred, average='weighted'), recall_score(y_test, y_test_pred, average='weighted'), f1_score(y_test, y_test_pred, average='weighted')]
        }
    return evaluation_results, metrics