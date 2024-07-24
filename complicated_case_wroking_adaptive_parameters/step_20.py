
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def step_20(Segments):
    Segments_normalized = []
    for segment in Segments:
        scaler = MinMaxScaler()
        normalized_segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
        Segments_normalized.append(normalized_segment)
    return Segments_normalized