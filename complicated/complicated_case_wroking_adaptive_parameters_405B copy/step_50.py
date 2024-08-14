import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score

def step_50(PCA_Features, kernel, nu, gamma):
    # Create labels
    labels = np.ones(len(PCA_Features))
    
    # Split data into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(PCA_Features, labels, test_size=0.2, random_state=42)
    
    # Create and fit a One-Class SVM classifier
    classifier = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    classifier.fit(train_features)
    
    # Predict labels for training data
    train_pred = classifier.predict(train_features)
    
    # Calculate error rate for training data
    train_error_rate = 1 - accuracy_score(train_labels, np.where(train_pred == -1, 0, 1))
    
    # Predict labels for test data (assume all test data as anomaly, i.e., -1)
    test_pred = classifier.predict(test_features)
    
    # Calculate error rate for test data
    test_error_rate = 1 - accuracy_score(np.ones(len(test_pred)) * -1, test_pred)
    
    # Calculate precision as 1 - error_rate for both training and test
    prec_learn = 1 - train_error_rate
    prec_test = 1 - test_error_rate
    
    return classifier, prec_learn, prec_test