import pandas as pd
import numpy as np

def step_31(df):
    df_cleaned = df.copy()
    data_types_info = df_cleaned.dtypes.apply(lambda x: x.name).to_dict()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    return df_cleaned, data_types_info