import pandas as pd
import numpy as np
from datetime import timedelta

def generate_synthetic_data(start_date, end_date, frequency='H'):
    """
    Generates synthetic time-series data for CPU usage, memory usage, and error rates.
    Simulates normal operation with occasional spikes or anomalies.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    n_points = len(date_range)

    # Simulate normal CPU usage (e.g., around 40-60%) with some noise
    cpu_usage = np.random.normal(loc=50, scale=5, size=n_points)
    cpu_usage = np.clip(cpu_usage, 0, 100) # Keep within 0-100%

    # Simulate normal Memory usage (e.g., around 60-80%) with some noise
    memory_usage = np.random.normal(loc=70, scale=7, size=n_points)
    memory_usage = np.clip(memory_usage, 0, 100) # Keep within 0-100%

    # Simulate normal error rates (e.g., very low, close to 0)
    error_rate = np.random.exponential(scale=0.1, size=n_points)
    error_rate = np.clip(error_rate, 0, 10) # Cap at 10% for realism

    # Introduce occasional anomalies (spikes)
    # CPU spike
    for _ in range(np.random.randint(2, 5)): # 2-5 spikes
        spike_idx = np.random.randint(0, n_points - 24) # Ensure spike doesn't go out of bounds for an hour
        cpu_usage[spike_idx:spike_idx+24] += np.random.uniform(30, 50)
        cpu_usage = np.clip(cpu_usage, 0, 100)

    # Memory spike
    for _ in range(np.random.randint(2, 5)): # 2-5 spikes
        spike_idx = np.random.randint(0, n_points - 24)
        memory_usage[spike_idx:spike_idx+24] += np.random.uniform(20, 40)
        memory_usage = np.clip(memory_usage, 0, 100)

    # Error rate spike
    for _ in range(np.random.randint(1, 3)): # 1-3 spikes
        spike_idx = np.random.randint(0, n_points - 24)
        error_rate[spike_idx:spike_idx+24] += np.random.uniform(5, 15)
        error_rate = np.clip(error_rate, 0, 100)

    df = pd.DataFrame({
        'timestamp': date_range,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'error_rate': error_rate
    })
    df = df.set_index('timestamp')
    return df

def create_labeled_incidents(df, cpu_threshold=85, memory_threshold=90, error_threshold=5, incident_duration_hours=6):
    """
    Labels incident intervals within the time-series data.
    An incident is triggered if any metric exceeds its threshold for a sustained period.
    """
    df['incident'] = 0

    # Detect potential anomaly points based on thresholds
    anomaly_cpu = df['cpu_usage'] > cpu_threshold
    anomaly_memory = df['memory_usage'] > memory_threshold
    anomaly_error = df['error_rate'] > error_threshold

    # Combine anomalies: an incident starts if any metric is anomalous
    potential_incident_points = anomaly_cpu | anomaly_memory | anomaly_error

    incident_start_times = df[potential_incident_points].index.tolist()

    labeled_incidents = []
    current_incident_end = None

    # Group close anomaly points into sustained incidents
    for ts in incident_start_times:
        if current_incident_end is None or ts > current_incident_end:
            # New incident starts
            incident_period_end = ts + timedelta(hours=incident_duration_hours)
            # Mark the incident period in the dataframe
            df.loc[ts : incident_period_end, 'incident'] = 1
            current_incident_end = incident_period_end
        else:
            # Extend current incident if anomaly point falls within existing incident window
            incident_period_end = ts + timedelta(hours=incident_duration_hours)
            if incident_period_end > current_incident_end:
                df.loc[current_incident_end : incident_period_end, 'incident'] = 1
                current_incident_end = incident_period_end

    return df

def create_sliding_window_samples(df, window_size, forecast_horizon):
    """
    Transforms time-series data into sliding window samples.
    Inputs: raw time-series data (DataFrame), window size W, forecast horizon H.
    Outputs: features (input sequences) and corresponding labels (future incident status).
    """
    features = []
    labels = []
    timestamps = []

    data = df[['cpu_usage', 'memory_usage', 'error_rate']].values
    incident_labels = df['incident'].values
    index_timestamps = df.index

    for i in range(len(data) - window_size - forecast_horizon + 1):
        # Extract the window features
        window_features = data[i : i + window_size]
        features.append(window_features.flatten()) # Flatten for a single row feature vector

        # Determine the label for the forecast horizon
        # An incident is true if any point within the forecast horizon has an incident
        horizon_labels = incident_labels[i + window_size : i + window_size + forecast_horizon]
        labels.append(1 if np.any(horizon_labels == 1) else 0)
        timestamps.append(index_timestamps[i + window_size - 1]) # Timestamp of the end of the window

    return np.array(features), np.array(labels), np.array(timestamps)

def split_data_by_time(features, labels, timestamps, split_ratio=0.8):
    """
    Splits the dataset by time into training and testing sets.
    Ensures temporal order of data is maintained.
    """
    split_idx = int(len(features) * split_ratio)

    X_train = features[:split_idx]
    y_train = labels[:split_idx]
    timestamps_train = timestamps[:split_idx]

    X_test = features[split_idx:]
    y_test = labels[split_idx:]
    timestamps_test = timestamps[split_idx:]

    return X_train, y_train, timestamps_train, X_test, y_test, timestamps_test
