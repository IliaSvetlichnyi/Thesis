import pandas as pd

def step_11(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: The file at '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv' contains parsing errors.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None