from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score

def step_41(features_train, features_test, labels_train, labels_test):
    model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
    model.fit(features_train)
    
    # Evaluate on training data
    train_predictions = model.predict(features_train)
    training_evaluation = accuracy_score(labels_train, train_predictions)
    
    # Evaluate on testing data
    test_predictions = model.predict(features_test)
    testing_evaluation = accuracy_score(labels_test, test_predictions)
    
    return model, training_evaluation, testing_evaluation
