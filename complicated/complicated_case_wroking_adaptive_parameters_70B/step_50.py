import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def step_50(PCA_Features, kernel, nu, gamma):
    # Create labels for learning data
    labels = np.ones(len(PCA_Features))

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test = train_test_split(PCA_Features, test_size=0.2, random_state=42)

    # Create and fit a One-Class SVM classifier
    FittedClassifier = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    FittedClassifier.fit(X_train)

    # Predict labels for training data
    y_pred_train = FittedClassifier.predict(X_train)

    # Calculate error rate for training data
    error_rate_train = np.mean(y_pred_train!= 1)
    
    # Predict labels for test data (all test data is assumed anomaly, i.e., -1)
    y_pred_test = FittedClassifier.predict(X_test)
    
    # Calculate error rate for test data
    error_rate_test = np.mean(y_pred_test == 1)
    
    # Calculate precision for training and test data
    Prec_learn = 1 - error_rate_train
    Prec_test = 1 - error_rate_test
    
    return FittedClassifier, Prec_learn, Prec_test