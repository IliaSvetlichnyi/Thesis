import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22

def step_31(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)  # Impute missing values with mean for numerical columns
    return df