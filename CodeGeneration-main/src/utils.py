"""
This module provides utility functions for data processing and step management
in the data pipeline application.
"""

import pandas as pd
from loguru import logger
import numpy as np
import json
from graph import Step

from typing import List, Dict


def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Extract and summarize information from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.

    Returns:
        Dict: A dictionary containing summarized information about the dataset,
              including columns, data types, sample data, value counts, and description.
    """
    columns = df.columns.tolist()
    types = df.dtypes.apply(lambda x: str(x)).to_dict()
    sample_data = df.head().to_dict(orient='list')
    value_counts = {col: df[col].value_counts().head().to_dict()
                    for col in df.columns}
    description = df.describe().to_dict()

    dataset_info = {
        'columns': columns,
        'types': types,
        'sample_data': sample_data,
        'value_counts': value_counts,
        'description': description
    }
    return dataset_info


def serialize_steps(steps: List[Step], path: str) -> None:
    """
    Serialize a list of Step objects to a JSON file.

    Args:
        steps (List[Step]): The list of Step objects to serialize.
        path (str): The file path where the serialized data will be saved.
    """
    with open(path, 'w') as f:
        data = [step.__dict__ for step in steps]
        json.dump(data, f, indent=4)


def deserialize_steps(path: str) -> List[Step]:
    """
    Deserialize a list of Step objects from a JSON file.

    Args:
        path (str): The file path of the JSON file containing serialized Step data.

    Returns:
        List[Step]: A list of deserialized Step objects.
    """
    steps = []
    with open(path, 'r') as f:
        data = json.load(f)
    for item in data:
        steps.append(Step(
            step_id=item['step_id'],
            description=item['description'],
            dependencies=item['dependencies'],
            input_vars=item['input_vars'],
            output_vars=item['output_vars'],
            additional_info=item['additional_info']
        ))
    return steps
