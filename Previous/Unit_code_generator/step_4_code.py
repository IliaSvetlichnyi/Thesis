import pandas as pd

df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

print(df.isnull().sum())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['age', 'bmi']] = imputer.fit_transform(df[['age', 'bmi']])