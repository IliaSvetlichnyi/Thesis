
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def step_61(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if model._estimator_type == 'classifier':
        metrics = ['accuracy', 'precision', 'recall', 'F1-score']
        evaluation_results = {
            'training': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, average='macro'),
                'recall': recall_score(y_train, y_train_pred, average='macro'),
                'F1-score': f1_score(y_train, y_train_pred, average='macro')
            },
            'testing': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred, average='macro'),
                'recall': recall_score(y_test, y_test_pred, average='macro'),
                'F1-score': f1_score(y_test, y_test_pred, average='macro')
            }
        }
    else:
        metrics = ['R^2', 'MSE', 'RMSE']
        evaluation_results = {
            'training': {
                'R^2': r2_score(y_train, y_train_pred),
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': mean_squared_error(y_train, y_train_pred, squared=False)
            },
            'testing': {
                'R^2': r2_score(y_test, y_test_pred),
                'MSE': mean_squared_error(y_test, y_test_pred),
                'RMSE': mean_squared_error(y_test, y_test_pred, squared=False)
            }
        }
    
    return evaluation_results, metrics