import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_31 import step_31
from step_41 import step_41

def main():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/complicated_case/learning-file_1.csv'
    
    # Step 11: Load and normalize the CSV file
    normalized_segments = step_11(csv_path)
    
    # Step 21: Extract features and perform PCA
    pca_features, pca = step_21(normalized_segments)
    
    # Step 31: Split the data into training and testing sets
    features_train, features_test, labels_train, labels_test = step_31(pca_features)
    
    # Step 41: Design, fit, and evaluate a One-Class SVM model on both training and testing data
    fitted_model, training_evaluation, testing_evaluation = step_41(features_train, features_test, labels_train, labels_test)
    
    print(f"Training evaluation: {training_evaluation}")
    print(f"Testing evaluation: {testing_evaluation}")

if __name__ == "__main__":
    main()
