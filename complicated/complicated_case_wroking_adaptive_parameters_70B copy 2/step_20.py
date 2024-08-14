

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def step_20(Segments):
    Segments_normalized = []
    for segment in Segments:
        scaler = MinMaxScaler()
        scaled_segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
        Segments_normalized.append(scaled_segment)
    return Segments_normalized