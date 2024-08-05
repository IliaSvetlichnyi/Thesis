
import pandas as pd
import numpy as np

def step_10(csv_path, SizeSegment=512):
    df = pd.read_csv(csv_path)
    signal = df['signal'].values
    num_segments = len(signal) // SizeSegment
    Segments = [signal[i*SizeSegment:(i+1)*SizeSegment] for i in range(num_segments)]
    return Segments