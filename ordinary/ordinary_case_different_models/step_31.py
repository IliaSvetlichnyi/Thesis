import pandas as pd

def step_31(df):
    df_cleaned = df.copy()
    data_types_info = {
        'age': 'int64',
        'sex': 'object',
        'bmi': 'float64',
        'children': 'int64',
        'smoker': 'object',
        'region': 'object',
        'charges': 'float64'
    }

    missing_values = df_cleaned.isnull().sum()
    if not missing_values.empty:
        df_cleaned.fillna(method='ffill', inplace=True)

    df_cleaned = df_cleaned.astype(data_types_info)

    return df_cleaned, data_types_info