from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def step_40(Features, NC_pca):
    """
    Apply PCA for dimension reduction.

    Parameters:
    Features (list): A list of feature arrays.
    NC_pca (int): The number of principal components to retain.

    Returns:
    PCA_Features (array): The transformed features.
    pca (PCA): The PCA object.
    """
    if not Features:
        raise ValueError("Features list is empty")
    if NC_pca <= 0:
        raise ValueError("NC_pca must be positive")

    pca = PCA(n_components=NC_pca)
    PCA_Features = pca.fit_transform(Features)

    return PCA_Features, pca