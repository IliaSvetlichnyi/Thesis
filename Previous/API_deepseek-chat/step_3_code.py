import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Display data types and missing values
print(df.info())

# Display statistical summary
print(df.describe(include='all'))

# Display value counts for categorical variables
print(df['sex'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())

# Display descriptive statistics for numerical variables
print(df['age'].describe())
print(df['bmi'].describe())
print(df['children'].describe())
print(df['charges'].describe())