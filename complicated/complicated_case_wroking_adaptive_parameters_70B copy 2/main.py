import pandas as pd

from step_10 import step_10
from step_20 import step_20
from step_30 import step_30
from step_40 import step_40
from step_50 import step_50

SizeSegment = 307
gamma = 'scale'
nu = 0.1
kernel = 'rbf'
NC_pca = 1
Dec_levels = 5

def main():
    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/complicated_case/learning-file_2.csv'
    Segments = step_10(csv_path, SizeSegment)
    Segments_normalized = step_20(Segments)
    Features = step_30(Segments_normalized, Dec_levels)
    PCA_Features, pca = step_40(Features, NC_pca)
    FittedClassifier, Prec_learn, Prec_test = step_50(PCA_Features, kernel, nu, gamma)
    print(f'Precision on training data: {Prec_learn:.2f}')
    print(f'Precision on test data: {Prec_test:.2f}')

if __name__ == '__main__':
    main()