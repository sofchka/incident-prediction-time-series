import argparse
import os
import numpy as np
import pandas as pd
from src.utils import generate_synthetic_data, create_labeled_incidents, create_sliding_window_samples, split_data_by_time

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic time-series data and process it for anomaly detection.")
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for data generation (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, default='2023-01-31', help='End date for data generation (YYYY-MM-DD).')
    parser.add_argument('--frequency', type=str, default='H', help='Frequency of data points (e.g., H for hourly, D for daily).')
    parser.add_argument('--window_size', type=int, default=24, help='Size of the sliding window (W).')
    parser.add_argument('--forecast_horizon', type=int, default=6, help='Forecast horizon (H).')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save the processed data.')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory '{args.output_dir}' ensured.")

    print("Generating synthetic time-series data...")
    df = generate_synthetic_data(args.start_date, args.end_date, args.frequency)
    print("Data generation complete. First 5 rows:")
    print(df.head())

    print("Creating labeled incidents...")
    df_labeled = create_labeled_incidents(df)
    print("Incident labeling complete. Incident distribution:")
    print(df_labeled['incident'].value_counts())

    print(f"Creating sliding window samples with window_size={args.window_size} and forecast_horizon={args.forecast_horizon}...")
    features, labels, timestamps = create_sliding_window_samples(df_labeled, args.window_size, args.forecast_horizon)
    print(f"Sliding window samples created. Features shape: {features.shape}, Labels shape: {labels.shape}")

    print("Splitting data into training and testing sets...")
    X_train, y_train, timestamps_train, X_test, y_test, timestamps_test = split_data_by_time(features, labels, timestamps)
    print(f"Data split complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Save processed data
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'timestamps_train.npy'), timestamps_train)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(args.output_dir, 'timestamps_test.npy'), timestamps_test)
    print(f"Processed data saved to '{args.output_dir}'.")

if __name__ == '__main__':
    main()
