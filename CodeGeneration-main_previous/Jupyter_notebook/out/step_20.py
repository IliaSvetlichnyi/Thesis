
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def step_20(Segments):
    Segments_normalized = []
    for segment in Segments:
        scaler = MinMaxScaler()
        normalized_segment = scaler.fit_transform(segment.reshape(-1, 1)).reshape(-1)
        Segments_normalized.append(normalized_segment)
    return Segments_normalized