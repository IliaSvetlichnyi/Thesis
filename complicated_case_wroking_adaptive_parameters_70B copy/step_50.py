
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

def step_50(PCA_Features, kernel, nu, gamma):
    labels = np.ones(len(PCA_Features))
    X_train, X_test = train_test_split(PCA_Features, test_size=0.2, random_state=42)
    clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    clf.fit(X_train)
    pred_train = clf.predict(X_train)
    err_train = np.mean(pred_train != 1)
    pred_test = clf.predict(X_test)
    err_test = np.mean(pred_test != -1)
    Prec_learn = 1 - err_train
    Prec_test = 1 - err_test
    return clf, Prec_learn, Prec_test