import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_31 import step_31
from step_32 import step_32
from step_51 import step_51

def validate_step():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
    df = step_11(csv_path)
    structure_info = step_21(df)
    df_cleaned, data_types_info = step_31(df)
    df_encoded = step_32(df_cleaned, data_types_info)
    model, X_train, X_test, y_train, y_test = step_51(df_encoded)
    print(model, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    validate_step()
