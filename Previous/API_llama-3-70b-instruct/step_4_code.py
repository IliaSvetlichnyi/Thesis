

import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Check for missing values
print(df.isnull().sum())

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[['age', 'bmi', 'children', 'charges']] = imputer.fit_transform(df[['age', 'bmi', 'children', 'charges']])

# Check again for missing values
print(df.isnull().sum())