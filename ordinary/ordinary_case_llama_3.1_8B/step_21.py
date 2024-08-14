

import pandas as pd
import numpy as np
from collections import Counter

def step_21(df):
    structure_info = {}
    structure_info['data_types'] = df.dtypes.to_dict()
    structure_info['column_names'] = df.columns.tolist()
    structure_info['value_counts'] = {}
    for col in df.columns:
        value_counts = Counter(df[col].value_counts().index)
        structure_info['value_counts'][col] = dict(value_counts.most_common(5))
    structure_info['statistical_description'] = df.describe().to_dict()
    return structure_info