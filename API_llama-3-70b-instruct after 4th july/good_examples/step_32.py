

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def step_32(df_cleaned, data_types_info):
    df_encoded = df_cleaned.copy()
    categorical_cols = [col for col, dtype in data_types_info.items() if dtype == 'object']
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded