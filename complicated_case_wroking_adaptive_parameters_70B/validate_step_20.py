import pandas as pd
from step_10 import step_10
from step_20 import step_20

SizeSegment = 307
gamma = 'scale'
nu = 0.1
kernel = 'rbf'
NC_pca = 1
Dec_levels = 5

def validate_step():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/complicated_case/learning-file_2.csv'
    Segments = step_10(csv_path, SizeSegment)
    Segments_normalized = step_20(Segments)
    print(Segments_normalized)

if __name__ == '__main__':
    validate_step()
