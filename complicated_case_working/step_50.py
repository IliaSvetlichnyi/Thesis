
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

def step_50(PCA_Features, kernel, nu, gamma):
    labels = np.ones((len(PCA_Features),))
    X_train, X_test = train_test_split(PCA_Features, test_size=0.2, random_state=42)
    clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    error_rate_train = np.mean(y_pred_train!= 1)
    y_pred_test = clf.predict(X_test)
    error_rate_test = np.mean(y_pred_test!= -1)
    Prec_learn = 1 - error_rate_train
    Prec_test = 1 - error_rate_test
    return clf, Prec_learn, Prec_test