import pandas as pd
from step_10 import step_10


csv_path = '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/complicated_case/learning-file_1.csv'
SizeSegment = 307
gamma = 'scale'
nu = 0.1
kernel = 'rbf'
NC_pca = 1
Dec_levels = 5

def validate_step():
    Segments = step_10(csv_path, SizeSegment)
    print(Segments)

if __name__ == '__main__':
    validate_step()
