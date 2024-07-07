# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

import pandas as pd

# Code from step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Load the dataset


# Code for current step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Assuming the dataset has already been loaded into a DataFrame named 'df'
pass

# Code from step 21: Examine the structure and characteristics of the data
# Load the dataset


# Code from step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Load the dataset


# Code for current step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Assuming the dataset has already been loaded into a DataFrame named 'df'
pass

# Code for current step 21: Examine the structure and characteristics of the data
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

# Code from step 22: Identify missing values, data types, and statistical summary
# Load the dataset


# Code from step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Load the dataset


# Code for current step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Assuming the dataset has already been loaded into a DataFrame named 'df'
pass

# Code from step 21: Examine the structure and characteristics of the data
# Load the dataset


# Code from step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Load the dataset


# Code for current step 11: Load the CSV file into a suitable format (e.g., DataFrame)
# Assuming the dataset has already been loaded into a DataFrame named 'df'
pass

# Code for current step 21: Examine the structure and characteristics of the data
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

# Code for current step 22: Identify missing values, data types, and statistical summary
# Identify missing values
missing_values = df.isnull().sum()

# Identify data types
data_types = df.dtypes

# Statistical summary
statistical_summary = df.describe(include='all')

# Display the results
missing_values, data_types, statistical_summary

# Code for current step 31: Handle missing values (remove or impute)
# Remove rows with any missing values
df_cleaned = df.dropna()

# Alternatively, impute missing values with the mean of each column
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)