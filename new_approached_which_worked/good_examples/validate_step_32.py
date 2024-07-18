import pandas as pd
from step_11 import step_11
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32

def validate_step():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
    df = step_11(csv_path)
    missing_values_info, data_types_info, stat_summary_info = step_22(df)
    df_cleaned = step_31(df, missing_values_info)
    df_encoded = step_32(df_cleaned, data_types_info)
    print(df_encoded)

if __name__ == '__main__':
    validate_step()
