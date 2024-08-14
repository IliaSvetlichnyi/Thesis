import pandas as pd

def step_11(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: Error parsing CSV file.")
        return None