
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def step_20(Segments):
    Segments_normalized = []
    for segment in Segments:
        scaler = MinMaxScaler()
        Segments_normalized.append(scaler.fit_transform(segment.reshape(-1, 1)).flatten())
    return Segments_normalized