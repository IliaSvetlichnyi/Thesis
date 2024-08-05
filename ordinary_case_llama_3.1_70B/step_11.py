

import pandas as pd

def step_11(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None