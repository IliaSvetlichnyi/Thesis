import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Display the first few rows of the dataset
print(data.head())

# Display the structure of the dataset
print(data.info())

# Display summary statistics
print(data.describe(include='all'))

# Display value counts for categorical variables
print(data['sex'].value_counts())
print(data['smoker'].value_counts())
print(data['region'].value_counts())

# Display descriptive statistics for numerical variables
print(data['age'].describe())
print(data['bmi'].describe())
print(data['children'].describe())
print(data['charges'].describe())