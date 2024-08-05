
import pandas as pd
import numpy as np

def step_31(df):
    df_cleaned = df.copy()
    data_types_info = df_cleaned.dtypes.apply(lambda x: x.name).to_dict()
    
    # Identify missing values
    missing_values_count = df_cleaned.isnull().sum()
    
    # Handle missing values
    for col in df_cleaned.columns:
        if missing_values_count[col] > 0:
            if data_types_info[col] == 'object':
                df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0], inplace=True)
            else:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    return df_cleaned, data_types_info