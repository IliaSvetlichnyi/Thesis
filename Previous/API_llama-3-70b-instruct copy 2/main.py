import pandas as pd

from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32
from step_35 import step_35
from step_51 import step_51
from step_52 import step_52
from step_53 import step_53
from step_61 import step_61
from step_62 import step_62

def main():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
    df = pd.read_csv(csv_path)
    df = step_11(df)
    df = step_21(df)
    df = step_22(df)
    df = step_31(df)
    df = step_32(df)
    df = step_35(df)
    df = step_51(df)
    df = step_52(df)
    df = step_53(df)
    df = step_61(df)
    df = step_62(df)

if __name__ == '__main__':
    main()