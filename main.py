import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the time-series anomaly detection project.")
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for data generation (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, default='2023-01-31', help='End date for data generation (YYYY-MM-DD).')
    parser.add_argument('--frequency', type=str, default='H', help='Frequency of data points (e.g., H for hourly, D for daily).')
    parser.add_argument('--window_size', type=int, default=24, help='Size of the sliding window (W).')
    parser.add_argument('--forecast_horizon', type=int, default=6, help='Forecast horizon (H).')
    parser.add_argument('--model_type', type=str, default='RandomForest', choices=['RandomForest', 'LogisticRegression'], help='Type of classifier to train.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in RandomForestClassifier (only for RandomForest).')
    parser.add_argument('--alert_threshold', type=float, default=0.5, help='Probability threshold for triggering an alert during evaluation.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

    args = parser.parse_args()

    # Define common directories
    data_dir = 'data'
    results_dir = 'results'

    # Get the directory of main.py, which is the project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Create an environment for subprocesses that includes the project root in PYTHONPATH
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = project_root + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = project_root

    print("\n--- Running Data Generation ---")
    generate_cmd = [
        'python',
        'src/generate_data.py',
        '--start_date', args.start_date,
        '--end_date', args.end_date,
        '--frequency', args.frequency,
        '--window_size', str(args.window_size),
        '--forecast_horizon', str(args.forecast_horizon),
        '--output_dir', data_dir
    ]
    subprocess.run(generate_cmd, check=True, env=env)
    print("--- Data Generation Complete ---")

    print("\n--- Running Model Training ---")
    train_cmd = [
        'python',
        'src/train.py',
        '--data_dir', data_dir,
        '--model_dir', results_dir,
        '--model_type', args.model_type,
        '--random_state', str(args.random_state)
    ]
    if args.model_type == 'RandomForest':
        train_cmd.extend(['--n_estimators', str(args.n_estimators)])
    subprocess.run(train_cmd, check=True, env=env)
    print("--- Model Training Complete ---")

    print("\n--- Running Model Evaluation ---")
    evaluate_cmd = [
        'python',
        'src/evaluate.py',
        '--data_dir', data_dir,
        '--model_dir', results_dir,
        '--results_dir', results_dir,
        '--alert_threshold', str(args.alert_threshold)
    ]
    subprocess.run(evaluate_cmd, check=True, env=env)
    print("--- Model Evaluation Complete ---")

    print("\n--- Project Orchestration Finished Successfully! ---")

if __name__ == '__main__':
    main()
