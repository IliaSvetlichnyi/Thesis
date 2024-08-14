import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_31 import step_31
from step_41 import step_41
from step_51 import step_51
from step_61 import step_61

csv_path = '/Users/ilya/Desktop/CodeGeneration-main/datasets/insurance.csv'

def main():
    df = step_11(csv_path)
    structure_info = step_21(df)
    df_cleaned, data_types_info = step_31(df)
    df_encoded = step_41(df_cleaned, data_types_info)
    model, X_train, X_test, y_train, y_test = step_51(df_encoded)
    evaluation_results, metrics = step_61(model, X_train, X_test, y_train, y_test)
    evaluation_results, metrics = step_61(model, X_train, X_test, y_train, y_test)
    print(evaluation_results, metrics)

if __name__ == '__main__':
    main()
