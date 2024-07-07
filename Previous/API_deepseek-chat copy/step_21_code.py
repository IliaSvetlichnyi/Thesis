import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Examine the structure and characteristics of the data
# Display the first few rows of the DataFrame
print(df.head())

# Get summary statistics of the DataFrame
print(df.describe())

# Display information about the DataFrame including data types and non-null values
print(df.info())

# Show the shape of the DataFrame (number of rows and columns)
print(df.shape)

# Display the data types of each column
print(df.dtypes)

# Check for missing values in the DataFrame
print(df.isnull().sum())