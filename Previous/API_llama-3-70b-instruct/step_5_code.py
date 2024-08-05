import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Convert categorical variables to numerical representations using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Print the resulting dataframe
print(df.head())