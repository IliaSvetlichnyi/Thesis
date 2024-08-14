import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_31 import step_31
from step_41 import step_41


csv_path = '/Users/ilya/Desktop/CodeGeneration-main/datasets/insurance.csv'

def validate_step():
    df = step_11(csv_path)
    structure_info = step_21(df)
    df_cleaned, data_types_info = step_31(df)
    df_encoded = step_41(df_cleaned, data_types_info)
    print(df_encoded)

if __name__ == '__main__':
    validate_step()
