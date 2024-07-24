import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def step_21(normalized_segments, n_components=0.95):
    features = []
    for segment in normalized_segments:
        df = pd.DataFrame(segment, columns=['value'])
        mean = df['value'].mean()
        std = df['value'].std()
        min_val = df['value'].min()
        max_val = df['value'].max()
        features.append([mean, std, min_val, max_val])
    
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    return pca_features, pca
