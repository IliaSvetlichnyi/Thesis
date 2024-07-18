

import pandas as pd

def step_11(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: Unable to parse the file. Please check the file format.")
        return None