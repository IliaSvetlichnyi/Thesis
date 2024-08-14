import numpy as np
from sklearn.preprocessing import MinMaxScaler

def step_20(Segments):
    Segments_normalized = []
    for segment in Segments:
        segment = segment.reshape(-1, 1)  # Reshape for MinMaxScaler
        scaler = MinMaxScaler()
        normalized_segment = scaler.fit_transform(segment).flatten()
        Segments_normalized.append(normalized_segment)
    return Segments_normalized