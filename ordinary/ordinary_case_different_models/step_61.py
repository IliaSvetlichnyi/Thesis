import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def step_61(model, X_train, X_test, y_train, y_test):
    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Initialize dictionary to hold evaluation metrics
    evaluation_results = {}
    metrics = []

    # Check if it's a classification task
    if hasattr(model, "classes_"):
        # Evaluation metrics for classification
        evaluation_results['Train Accuracy'] = accuracy_score(y_train, y_train_pred)
        evaluation_results['Test Accuracy'] = accuracy_score(y_test, y_test_pred)
        evaluation_results['Train Precision'] = precision_score(y_train, y_train_pred, average='weighted')
        evaluation_results['Test Precision'] = precision_score(y_test, y_test_pred, average='weighted')
        evaluation_results['Train Recall'] = recall_score(y_train, y_train_pred, average='weighted')
        evaluation_results['Test Recall'] = recall_score(y_test, y_test_pred, average='weighted')
        evaluation_results['Train F1-score'] = f1_score(y_train, y_train_pred, average='weighted')
        evaluation_results['Test F1-score'] = f1_score(y_test, y_test_pred, average='weighted')
        metrics = ['accuracy', 'precision', 'recall', 'F1-score']

    else:
        # Evaluation metrics for regression
        evaluation_results['Train R^2'] = r2_score(y_train, y_train_pred)
        evaluation_results['Test R^2'] = r2_score(y_test, y_test_pred)
        evaluation_results['Train MSE'] = mean_squared_error(y_train, y_train_pred)
        evaluation_results['Test MSE'] = mean_squared_error(y_test, y_test_pred)
        evaluation_results['Train RMSE'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        evaluation_results['Test RMSE'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics = ['R^2', 'MSE', 'RMSE']

    # Print evaluation results
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")

    return evaluation_results, metrics