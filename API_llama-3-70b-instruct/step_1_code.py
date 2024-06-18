import pandas as pd

file_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.info())
print(df.describe())
print(df['age'].value_counts())
print(df['sex'].value_counts())
print(df['bmi'].value_counts())
print(df['children'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())
print(df['charges'].value_counts())