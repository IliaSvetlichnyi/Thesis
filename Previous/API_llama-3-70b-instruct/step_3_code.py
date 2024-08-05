
import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Identify missing values
print(df.isnull().sum())

# Identify data types
print(df.dtypes)

# Statistical summary
print(df.describe())

# Value counts for each column
print(df['age'].value_counts())
print(df['sex'].value_counts())
print(df['bmi'].value_counts())
print(df['children'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())
print(df['charges'].value_counts())