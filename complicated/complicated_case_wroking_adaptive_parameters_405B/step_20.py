from sklearn.preprocessing import MinMaxScaler
import numpy as np

def step_20(Segments):
    Segments_normalized = []
    for segment in Segments:
        scaler = MinMaxScaler()
        segment_normalized = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
        Segments_normalized.append(segment_normalized)
    return Segments_normalized