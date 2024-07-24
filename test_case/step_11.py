import csv
import pandas as pd

def step_11(csv_path, size=512):
    segments = []
    with open(csv_path, newline='') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        window = []
        for i, row in enumerate(csvreader):
            window.append(float(row[1]))
            if (i + 1) % size == 0:
                segments.append(window)
                window = []
        if window:
            segments.append(window)
    
    normalized_segments = []
    for segment in segments:
        df = pd.DataFrame(segment, columns=['value'])
        min_val = df['value'].min()
        max_val = df['value'].max()
        normalized_segment = (df['value'] - min_val) / (max_val - min_val)
        normalized_segments.append(normalized_segment.values.tolist())
    return normalized_segments
