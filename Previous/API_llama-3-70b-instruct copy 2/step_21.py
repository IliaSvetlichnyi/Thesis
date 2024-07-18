import pandas as pd
from step_11 import step_11

def step_21(df):
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.nunique())
    for col in df.select_dtypes(include=['object']).columns:
        print(df[col].value_counts().head(5))
    return df