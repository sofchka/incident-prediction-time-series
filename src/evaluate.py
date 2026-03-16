import argparse
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained anomaly detection model.")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing processed test data (X_test.npy, y_test.npy).')
    parser.add_argument('--model_dir', type=str, default='results', help='Directory containing the trained model (model.joblib).')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save evaluation results and plots.')
    parser.add_argument('--alert_threshold', type=float, default=0.5, help='Probability threshold for triggering an alert.')

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results directory '{args.results_dir}' ensured.")

    print(f"Loading test data from '{args.data_dir}'...")
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    timestamps_test = np.load(os.path.join(args.data_dir, 'timestamps_test.npy'), allow_pickle=True)
    print(f"Test data loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print(f"Loading trained model from '{args.model_dir}'...")
    model_path = os.path.join(args.model_dir, 'model.joblib')
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    print("Making predictions...")
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class
    else:
        # For models without predict_proba (e.g., SVC with probability=False), y_proba can be y_pred
        y_proba = y_pred

    # --- Standard Classification Metrics ---
    print("\n--- Standard Classification Metrics ---")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # --- Metrics with Alert Threshold ---
    print(f"\n--- Metrics with Alert Threshold ({args.alert_threshold}) ---")
    alert_predictions = (y_proba >= args.alert_threshold).astype(int)
    alert_precision = precision_score(y_test, alert_predictions)
    alert_recall = recall_score(y_test, alert_predictions)
    alert_f1 = f1_score(y_test, alert_predictions)

    print(f"Alert Precision (threshold={args.alert_threshold}): {alert_precision:.4f}")
    print(f"Alert Recall (threshold={args.alert_threshold}): {alert_recall:.4f}")
    print(f"Alert F1-Score (threshold={args.alert_threshold}): {alert_f1:.4f}")

    # --- ROC Curve Plot ---
    print("\nGenerating ROC Curve plot...")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(args.results_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close() # Close plot to free up memory
    print(f"ROC Curve saved to '{roc_curve_path}'.")

    # --- Save results to a text file ---
    results_summary_path = os.path.join(args.results_dir, 'evaluation_summary.txt')
    with open(results_summary_path, 'w') as f:
        f.write("--- Evaluation Summary ---\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Data Directory: {args.data_dir}\n")
        f.write(f"Model Directory: {args.model_dir}\n")
        f.write(f"Alert Threshold: {args.alert_threshold}\n\n")

        f.write("--- Standard Classification Metrics ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")

        f.write("--- Confusion Matrix ---\n")
        f.write(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}\n")
        f.write(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}\n\n")

        f.write(f"--- Metrics with Alert Threshold ({args.alert_threshold}) ---\n")
        f.write(f"Alert Precision: {alert_precision:.4f}\n")
        f.write(f"Alert Recall: {alert_recall:.4f}\n")
        f.write(f"Alert F1-Score: {alert_f1:.4f}\n\n")

        f.write("--- Classification Report ---\n")
        f.write(classification_report(y_test, y_pred))

    print(f"Evaluation summary saved to '{results_summary_path}'.")

if __name__ == '__main__':
    main()
