import pandas as pd
import numpy as np

def step_10(csv_path, SizeSegment):
    """
    Import raw data from CSV and segment it.

    Parameters:
    csv_path (str): Path to the CSV file containing the raw data.
    SizeSegment (int): Size of each segment.

    Returns:
    Segments (list): List of 1D numpy arrays, each representing a segment of the signal data.
    """
    # Check if the CSV file is empty
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    # Ensure the 'signal' column is of type int64
    df['signal'] = df['signal'].astype(np.int64)

    # Create segments of size SizeSegment
    segments = [df['signal'].values[i:i+SizeSegment] for i in range(0, len(df), SizeSegment)]

    # Remove any incomplete segments at the end
    if len(segments) > 0 and len(segments[-1]) < SizeSegment:
        segments = segments[:-1]

    return segments