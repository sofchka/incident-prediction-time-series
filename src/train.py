import argparse
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser(description="Train a baseline classifier for anomaly detection.")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing processed data (X_train.npy, y_train.npy).')
    parser.add_argument('--model_dir', type=str, default='results', help='Directory to save the trained model.')
    parser.add_argument('--model_type', type=str, default='RandomForest', choices=['RandomForest', 'LogisticRegression'], help='Type of classifier to train.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in RandomForestClassifier.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

    args = parser.parse_args()

    # Create model output directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"Model output directory '{args.model_dir}' ensured.")

    print(f"Loading training data from '{args.data_dir}'...")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    print(f"Training data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    print(f"Initializing {args.model_type} classifier...")
    if args.model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, class_weight='balanced')
    elif args.model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=args.random_state, solver='liblinear', class_weight='balanced')
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    print("Training classifier...")
    model.fit(X_train, y_train)
    print("Classifier training complete.")

    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Trained model saved to '{model_path}'.")

if __name__ == '__main__':
    main()
