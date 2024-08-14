import pandas as pd
import numpy as np

def step_10(csv_path, SizeSegment):
    data = pd.read_csv(csv_path)
    if 'timestamp' not in data.columns or 'signal' not in data.columns:
        raise ValueError('Expected columns "timestamp" and "signal" in the CSV file')
    if SizeSegment <= 0:
        raise ValueError('SizeSegment must be a positive integer')
    
    signal_data = data['signal'].values
    num_segments = len(signal_data) // SizeSegment
    remaining_samples = len(signal_data) % SizeSegment
    
    Segments = [signal_data[i*SizeSegment:(i+1)*SizeSegment] for i in range(num_segments)]
    
    if remaining_samples > 0:
        last_segment = np.pad(signal_data[num_segments*SizeSegment:], (0, SizeSegment-remaining_samples), mode='constant')
        Segments.append(last_segment)
    
    return Segments