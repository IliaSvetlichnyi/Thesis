

import pandas as pd
from sklearn.decomposition import PCA

def step_40(Features, NC_pca):
    pca = PCA(n_components=NC_pca)
    PCA_Features = pca.fit_transform(Features)
    return PCA_Features, pca