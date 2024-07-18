import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def step_61(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if y_train.dtype.kind in {'i', 'f'}:  # check if target variable is numerical (regression)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = train_mse ** 0.5
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = test_mse ** 0.5
        test_r2 = r2_score(y_test, y_test_pred)
        evaluation_results = {
            'Train': {'R^2': train_r2, 'MSE': train_mse, 'RMSE': train_rmse},
            'Test': {'R^2': test_r2, 'MSE': test_mse, 'RMSE': test_rmse}
        }
        metrics = ['R^2', 'MSE', 'RMSE']
    else:  # target variable is categorical (classification)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        evaluation_results = {
            'Train': {'Accuracy': train_accuracy, 'Precision': train_precision, 'Recall': train_recall, 'F1-score': train_f1},
            'Test': {'Accuracy': test_accuracy, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1}
        }
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    return evaluation_results, metrics