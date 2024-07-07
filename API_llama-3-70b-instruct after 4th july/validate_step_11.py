import pandas as pd
from step_11 import step_11

def validate_step():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
    df = step_11(csv_path)
    print(df)

if __name__ == '__main__':
    validate_step()
