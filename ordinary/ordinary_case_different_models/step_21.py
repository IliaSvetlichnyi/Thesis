import pandas as pd

def step_21(df):
    structure_info = {}

    structure_info['columns'] = list(df.columns)
    structure_info['data_types'] = df.dtypes.to_dict()

    structure_info['value_counts'] = {col: df[col].value_counts().head(5).to_dict() for col in df.columns}
    structure_info['statistical_description'] = df.describe(include='all').to_dict()

    return structure_info