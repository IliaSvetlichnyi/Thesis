from sklearn.decomposition import PCA

def step_40(Features, NC_pca):
    """
    Apply PCA for dimension reduction.
    
    Parameters:
    Features (array-like): The input features.
    NC_pca (int): The number of components for PCA.

    Returns:
    PCA_Features (array-like): The transformed features.
    pca (PCA): The PCA object.
    """
    if not isinstance(Features, (list, tuple)) or not all(isinstance(x, (list, tuple)) for x in Features):
        raise ValueError("Features must be a list of lists or tuples")

    if NC_pca <= 0:
        raise ValueError("NC_pca must be a positive integer")

    pca = PCA(n_components=NC_pca)
    try:
        PCA_Features = pca.fit_transform(Features)
    except ValueError as e:
        raise ValueError("Failed to apply PCA: {}".format(e))

    return PCA_Features, pca