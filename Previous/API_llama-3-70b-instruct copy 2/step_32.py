import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31

def step_32(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df