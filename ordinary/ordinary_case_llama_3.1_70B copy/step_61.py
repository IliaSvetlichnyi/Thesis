

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def step_61(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if y_train.dtype == 'object':
        evaluation_results = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, average='macro'),
                'recall': recall_score(y_train, y_train_pred, average='macro'),
                'f1-score': f1_score(y_train, y_train_pred, average='macro')
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, average='macro'),
                'recall': recall_score(y_test, y_test_pred, average='macro'),
                'f1-score': f1_score(y_test, y_test_pred, average='macro')
            }
        }
        metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    else:
        evaluation_results = {
            'train': {
                'R^2': r2_score(y_train, y_train_pred),
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': mean_squared_error(y_train, y_train_pred, squared=False)
            },
            'test': {
                'R^2': r2_score(y_test, y_test_pred),
                'MSE': mean_squared_error(y_test, y_test_pred),
                'RMSE': mean_squared_error(y_test, y_test_pred, squared=False)
            }
        }
        metrics = ['R^2', 'MSE', 'RMSE']

    print("Evaluation Results:")
    for dataset, results in evaluation_results.items():
        print(f"{dataset.capitalize()}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        print()

    return evaluation_results, metrics