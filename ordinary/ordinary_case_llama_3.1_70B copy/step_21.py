

import pandas as pd

def step_21(df):
    structure_info = {
        'data_types': df.dtypes.to_dict(),
        'value_counts': {col: df[col].value_counts().head(5).to_dict() for col in df.columns},
        'statistical_description': df.describe().to_dict()
    }
    return structure_info