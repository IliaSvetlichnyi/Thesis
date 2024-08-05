import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_31 import step_31

def validate_step():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv'
    df = step_11(csv_path)
    structure_info = step_21(df)
    df_cleaned, data_types_info = step_31(df)
    print(df_cleaned, data_types_info)

if __name__ == '__main__':
    validate_step()
