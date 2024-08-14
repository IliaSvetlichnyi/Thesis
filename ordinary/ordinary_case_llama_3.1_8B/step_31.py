

import pandas as pd
import numpy as np
import scipy.stats as stats

def step_31(df):
    df_cleaned = df.copy()
    data_types_info = {col: df[col].dtype for col in df.columns}
    
    # Identify missing values
    missing_values = df.isnull().sum()
    if len(missing_values[missing_values > 0]) > 0:
        for col, count in missing_values[missing_values > 0].items():
            if data_types_info[col] == 'int64':
                df_cleaned[col] = df_cleaned[col].fillna(df[col].median())
            elif data_types_info[col] == 'float64':
                df_cleaned[col] = df_cleaned[col].fillna(df[col].mean())
            elif data_types_info[col] == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df[col].mode()[0])
    
    return df_cleaned, data_types_info