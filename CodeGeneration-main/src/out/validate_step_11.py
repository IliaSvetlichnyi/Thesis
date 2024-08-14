import pandas as pd
from step_11 import step_11


csv_path = '/Users/ilya/Desktop/CodeGeneration-main/datasets/insurance.csv'

def validate_step():
    df = step_11(csv_path)
    print(df)

if __name__ == '__main__':
    validate_step()
