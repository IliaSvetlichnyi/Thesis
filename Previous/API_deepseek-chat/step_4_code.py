import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"
data = pd.read_csv(file_path)

# Check for missing values
print(data.isnull().sum())

# Handle missing values
# Since the dataset does not have missing values, we'll use a placeholder for demonstration
# Let's assume 'bmi' has missing values and we want to impute them with the mean

# Create an imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the 'bmi' column
data['bmi'] = imputer.fit_transform(data[['bmi']])

# Verify that missing values are handled
print(data.isnull().sum())