

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def step_50(PCAN_Features, kernel, nu, gamma):
    Labels = np.ones(len(PCAN_Features))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(PCAN_Features, Labels, test_size=0.2, random_state=42)

    FittedClassifier = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    FittedClassifier.fit(X_train)

    y_pred_train = FittedClassifier.predict(X_train)
    error_rate_train = np.mean(y_pred_train!= 1)
    Prec_learn = 1 - error_rate_train

    y_pred_test = FittedClassifier.predict(X_test)
    error_rate_test = np.mean(y_pred_test!= -1)
    Prec_test = 1 - error_rate_test

    return FittedClassifier, Prec_learn, Prec_test