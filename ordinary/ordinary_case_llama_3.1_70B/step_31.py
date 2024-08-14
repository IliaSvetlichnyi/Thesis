import pandas as pd
import numpy as np

def step_31(df):
    # Identify missing values
    missing_values = df.isnull().sum()
    
    # Identify data types
    data_types_info = df.dtypes.to_dict()
    
    # Handle missing values if there are any
    df_cleaned = df.copy()
    for col in df.columns:
        if missing_values[col] > 0:
            if data_types_info[col] == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    
    return df_cleaned, data_types_info