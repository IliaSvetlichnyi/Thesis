import pandas as pd
import numpy as np

def read_csv_fill_missing_print(csv_path):
    df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/Iris.csv')
    df.fillna(df.mean(), inplace=True)
    print(df.head(5))