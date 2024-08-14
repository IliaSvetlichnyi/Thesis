import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

def step_50(PCA_Features, kernel='rbf', nu=0.1, gamma='scale'):
    # Create labels
    labels = np.ones(len(PCA_Features))
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(PCA_Features, labels, test_size=0.2, random_state=42)
    
    # Create and fit a One-Class SVM classifier
    oc_svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    oc_svm.fit(X_train)
    
    # Predict labels for training data
    y_train_pred = oc_svm.predict(X_train)
    train_error_rate = np.mean(y_train_pred != 1)
    
    # Predict labels for test data (assume all test data as anomaly, i.e., -1)
    y_test_pred = oc_svm.predict(X_test)
    test_error_rate = np.mean(y_test_pred != -1)
    
    # Calculate precision for both training and test sets
    Prec_learn = 1 - train_error_rate
    Prec_test = 1 - test_error_rate
    
    return oc_svm, Prec_learn, Prec_test