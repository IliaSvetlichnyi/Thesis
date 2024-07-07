import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Check missing values
print(df.isna().sum())

# Get data types
print(df.dtypes)

# Calculate summary statistics
print(df.describe())