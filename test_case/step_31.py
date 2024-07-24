import pandas as pd
from sklearn.model_selection import train_test_split

def step_31(pca_features, test_size=0.2, random_state=42):
    pca_df = pd.DataFrame(pca_features)
    labels = pd.Series([1] * len(pca_features))  # All samples are normal (label 1)
    features_train, features_test, labels_train, labels_test = train_test_split(pca_df, labels, test_size=test_size, random_state=random_state)
    return features_train, features_test, labels_train, labels_test
