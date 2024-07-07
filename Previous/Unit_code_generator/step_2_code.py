import pandas as pd
import numpy as np

data = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

print(data.head())
print(data.info())
print(data.describe())