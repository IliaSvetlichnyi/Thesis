

import pandas as pd
import pywt
import numpy as np

def step_20(Segments):
    return np.array(Segments)  # implementation of step_20 function

def step_30(Segments_normalized, Dec_levels):
    Features = []
    for segment in Segments_normalized:
        coeffs = pywt.wavedec(segment, 'db3', level=int(Dec_levels))
        features = [coefficient.mean() for coefficient in coeffs]
        Features.append(features)
    return Features

Segments = [1, 2, 3, 4, 5]  # define Segments here
Dec_levels = 2  # define Dec_levels here
Segments_normalized = step_20(Segments)
Features = step_30(Segments_normalized, Dec_levels)