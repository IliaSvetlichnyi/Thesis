

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def step_10(csv_path, SizeSegment):
    data = pd.read_csv(csv_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='ISO8601')
    signal = data['signal'].values
    Segments = [signal[i:i + SizeSegment] for i in range(0, len(signal), SizeSegment)]
    Segments_normalized = [MinMaxScaler().fit_transform(segment.reshape(-1, 1)).ravel() for segment in Segments]
    return Segments_normalized