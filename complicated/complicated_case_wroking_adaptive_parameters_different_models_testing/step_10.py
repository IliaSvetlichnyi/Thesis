import pandas as pd
import numpy as np

def step_10(csv_path, SizeSegment):
    data = pd.read_csv(csv_path)
    segments = [data['signal'][i:i + SizeSegment].values for i in range(0, len(data), SizeSegment)]
    segments = [segment for segment in segments if len(segment) == SizeSegment]
    return segments