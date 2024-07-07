

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def step_61(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    evaluation_results = {}

    if y_train.dtype.kind == 'i' or y_train.dtype.kind == 'f':
        # regression
        evaluation_results['train_R2'] = r2_score(y_train, y_pred_train)
        evaluation_results['test_R2'] = r2_score(y_test, y_pred_test)
        evaluation_results['train_MSE'] = mean_squared_error(y_train, y_pred_train)
        evaluation_results['test_MSE'] = mean_squared_error(y_test, y_pred_test)
        evaluation_results['train_RMSE'] = evaluation_results['train_MSE'] ** 0.5
        evaluation_results['test_RMSE'] = evaluation_results['test_MSE'] ** 0.5
        metrics = ['R2', 'MSE', 'RMSE']
    else:
        # classification
        evaluation_results['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        evaluation_results['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        evaluation_results['train_precision'] = precision_score(y_train, y_pred_train, average='macro')
        evaluation_results['test_precision'] = precision_score(y_test, y_pred_test, average='macro')
        evaluation_results['train_recall'] = recall_score(y_train, y_pred_train, average='macro')
        evaluation_results['test_recall'] = recall_score(y_test, y_pred_test, average='macro')
        evaluation_results['train_f1'] = f1_score(y_train, y_pred_train, average='macro')
        evaluation_results['test_f1'] = f1_score(y_test, y_pred_test, average='macro')
        metrics = ['accuracy', 'precision', 'recall', 'f1']

    return evaluation_results, metrics