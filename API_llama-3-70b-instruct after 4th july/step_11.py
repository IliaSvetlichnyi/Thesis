def step_11(csv_path):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print("File not found. Please check the csv_path.")
        return None