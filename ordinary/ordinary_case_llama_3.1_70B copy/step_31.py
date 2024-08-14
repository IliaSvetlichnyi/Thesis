import pandas as pd
import numpy as np

def step_31(df):
    data_types_info = df.dtypes.apply(lambda x: x.name).to_dict()
    df_cleaned = df.copy()
    
    for col in df.columns:
        if data_types_info[col] == 'object':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            
    return df_cleaned, data_types_info