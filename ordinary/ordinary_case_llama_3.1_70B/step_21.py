import pandas as pd

def step_21(df):
    """
    Examine the structure and characteristics of the data.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be examined.

    Returns:
    structure_info (dict): A dictionary containing the structure and characteristics of the data.
    """
    data_types_info = df.dtypes.apply(lambda x: x.name).to_dict()
    data_sample = df.head()
    column_value_counts = df.apply(lambda x: x.value_counts().head(5).to_dict())
    statistical_description = df.describe().to_dict()

    structure_info = {
        'column_names': list(df.columns),
        'data_types': data_types_info,
        'sample_data': data_sample,
        'value_counts': column_value_counts,
        'statistical_description': statistical_description
    }

    return structure_info