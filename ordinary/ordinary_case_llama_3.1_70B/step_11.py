import pandas as pd

def step_11(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if not df.shape[0]:
            raise ValueError("Empty DataFrame")
        expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError("Missing expected columns in DataFrame")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv'")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty CSV file")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file")