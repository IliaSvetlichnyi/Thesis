import pandas as pd
import pywt

def step_30(Segments_normalized, Dec_levels):
    Features = []
    for segment in Segments_normalized:
        coeffs = pywt.wavedec(segment, 'db3', level=Dec_levels)
        features = [coefficient.mean() for coefficient in coeffs]
        Features.append(features)
    return Features