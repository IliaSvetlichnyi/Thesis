

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def step_41(df_cleaned, data_types_info):
    df_encoded = df_cleaned.copy()
    categorical_cols = [col for col, dtype in data_types_info.items() if dtype == 'object']
    encoder = OrdinalEncoder()
    df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])
    return df_encoded