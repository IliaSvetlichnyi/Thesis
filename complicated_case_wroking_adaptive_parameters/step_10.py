
import pandas as pd
import numpy as np

def step_10(csv_path, SizeSegment):
    df = pd.read_csv(csv_path)
    signal_values = df['signal'].values
    num_segments = len(signal_values) // SizeSegment
    Segments = [signal_values[i * SizeSegment:(i + 1) * SizeSegment] for i in range(num_segments)]
    return Segments