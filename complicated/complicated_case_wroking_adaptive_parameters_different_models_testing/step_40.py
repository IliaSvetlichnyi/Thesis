import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def step_40(Features, NC_pca):
    if not isinstance(Features, np.ndarray):
        Features = np.array(Features)
        
    if Features.size == 0:
        raise ValueError("Features array cannot be empty.")
        
    if NC_pca < 1 or NC_pca > Features.shape[1]:
        raise ValueError("NC_pca should be in the range [1, number of features]")

    scaler = StandardScaler()
    Features_scaled = scaler.fit_transform(Features)
    
    pca = PCA(n_components=NC_pca)
    PCA_Features = pca.fit_transform(Features_scaled)
    
    return PCA_Features, pca