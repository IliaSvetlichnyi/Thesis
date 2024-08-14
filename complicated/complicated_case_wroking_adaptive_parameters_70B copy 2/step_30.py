
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler

def step_30(Segments, Dec_levels):
    Segments_normalized = [MinMaxScaler().fit_transform(segment.reshape(-1, 1))[:, 0] for segment in Segments]
    Features = []
    for segment in Segments_normalized:
        coeffs = pywt.wavedec(segment, 'db3', level=Dec_levels)
        features = [coefficient.mean() for coefficient in coeffs]
        Features.append(features)
    return Features