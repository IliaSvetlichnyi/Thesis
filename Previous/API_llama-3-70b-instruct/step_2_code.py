
import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Print the data types and summary statistics
print(df.info())
print(df.describe())

# Print the value counts for each column
for col in df.columns:
    print(df[col].value_counts())