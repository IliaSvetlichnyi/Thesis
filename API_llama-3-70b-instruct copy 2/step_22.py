import pandas as pd
from step_11 import step_11
from step_21 import step_21



import pandas as pd
import numpy as np

def step_22(df):
    # Identify missing values
    missing_values_count = df.isnull().sum()
    print("Missing values count: \n", missing_values_count)
    
    # Identify data types
    data_types = df.dtypes
    print("Data types: \n", data_types)
    
    # Statistical summary
    statistical_summary = df.describe()
    print("Statistical summary: \n", statistical_summary)
    
    return df