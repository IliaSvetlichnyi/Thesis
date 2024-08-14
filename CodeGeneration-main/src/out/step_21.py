

import pandas as pd

def step_21(df):
    structure_info = {
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'value_counts': {column: df[column].value_counts().head(5).to_dict() for column in df.columns},
        'statistical_description': df.describe().to_dict()
    }
    return structure_info