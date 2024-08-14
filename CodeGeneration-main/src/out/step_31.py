

import pandas as pd
import numpy as np

def step_31(df):
    # Identify missing values
    missing_values = df.isnull().sum()
    print("Missing values:", missing_values)
    
    # Identify data types
    data_types = df.dtypes
    print("Data types:", data_types)
    
    # Handle missing values
    if missing_values.any():
        df_cleaned = df.fillna(df.mean())  # replace missing values with mean
    else:
        df_cleaned = df
    
    data_types_info = data_types.to_dict()
    return df_cleaned, data_types_info