# Time-Series Anomaly Detection Project

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Methodology](#2-methodology)
    *   [2.1. Synthetic Data Generation](#21-synthetic-data-generation)
    *   [2.2. Sliding Window Formulation (W and H)](#22-sliding-window-formulation-w-and-h)
    *   [2.3. Incident Labeling](#23-incident-labeling)
    *   [2.4. Model Selection](#24-model-selection)
3.  [Project Structure](#3-project-structure)
4.  [How to Run the Project](#4-how-to-run-the-project)
5.  [Evaluation Metrics and Results](#5-evaluation-metrics-and-results)
6.  [Analysis of Results](#6-analysis-of-results)
7.  [Limitations and Future Work](#7-limitations-and-future-work)
8.  [Real-World Adaptation](#8-real-world-adaptation)

## 1. Introduction
Time-series anomaly detection is a critical task in many domains, including system monitoring, fraud detection, and industrial control. Anomalies, or 'incidents,' in time-series data often indicate significant events such as system failures, cyber-attacks, or operational deviations that require immediate attention. This project develops a framework for detecting such anomalies in synthetic system metrics (CPU usage, memory usage, error rates) using a machine learning approach.

The goal is to build a robust system that can identify potential incidents by learning patterns from historical data and flagging unusual behavior that deviates from the norm.

## 2. Methodology
Our approach involves generating synthetic time-series data, formulating the anomaly detection problem as a supervised learning task using a sliding window, training a baseline classifier, and evaluating its performance.

### 2.1. Synthetic Data Generation
Synthetic time-series data for CPU usage, memory usage, and error rates is generated to simulate normal system behavior with occasional anomalies (spikes). The `generate_synthetic_data` function in `src/utils.py` creates data points at a specified frequency. Normal behavior is modeled using Gaussian and exponential distributions. Anomalies are introduced by adding significant spikes to these metrics at random intervals, simulating real-world incidents.

### 2.2. Sliding Window Formulation (W and H)
The problem is transformed into a supervised learning task using a sliding window approach. For each point in time, a fixed-size window of past observations becomes the input features, and a fixed-size window into the future determines the label.

*   **W (Window Size)**: Represents the number of past time steps to consider as input features for predicting future incidents. For example, if `W=24` (hourly data), the model uses the last 24 hours of CPU, memory, and error rate data to make a prediction.
*   **H (Forecast Horizon)**: Represents the number of future time steps to predict for incident occurrence. If `H=6`, the model predicts whether an incident will occur within the next 6 hours after the current window. A label of '1' indicates an incident in the forecast horizon, while '0' indicates no incident.

The `create_sliding_window_samples` function in `src/utils.py` handles this transformation.

### 2.3. Incident Labeling
Incidents are labeled based on predefined thresholds for CPU usage, memory usage, and error rates. If any metric exceeds its threshold for a sustained period (e.g., 6 hours), that period is marked as an 'incident'. The `create_labeled_incidents` function in `src/utils.py` applies this logic to generate the ground truth labels for the synthetic data.

### 2.4. Model Selection
For this baseline project, we chose standard machine learning classifiers:
*   **RandomForestClassifier**: An ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It's robust to overfitting and can handle non-linear relationships.
*   **LogisticRegression**: A linear model used for binary classification. While simpler, it provides a strong baseline and is interpretable.

These models are suitable for a first pass at anomaly detection due to their interpretability and relatively good performance on tabular data.

## 3. Project Structure
```
. # Project Root
├── main.py             # Orchestrates the entire pipeline (data gen, train, eval)
├── requirements.txt    # List of Python dependencies
├── README.md           # This file
├── data/               # Directory for processed data (X_train.npy, y_train.npy, etc.)
├── results/            # Directory for trained models and evaluation outputs
└── src/
    ├── __init__.py     # Makes `src` a Python package
    ├── utils.py        # Helper functions (data generation, windowing, labeling)
    ├── generate_data.py # Script to generate and process synthetic data
    ├── train.py        # Script to train a classifier
    └── evaluate.py     # Script to evaluate the trained model
```

## 4. How to Run the Project
Follow these steps to set up and run the anomaly detection project:

1.  **Clone the repository (if applicable):**
    ```bash
    # Assuming you have a git repository
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main orchestration script:**
    The `main.py` script will execute the entire pipeline: data generation, model training, and evaluation.

    **Basic execution:**
    ```bash
    python main.py
    ```

    **Execution with custom parameters:**
    You can customize various parameters using command-line arguments. For example:
    ```bash
    python main.py --start_date 2023-01-01 --end_date 2023-03-31 --frequency H \
                   --window_size 48 --forecast_horizon 12 --model_type LogisticRegression \
                   --alert_threshold 0.6
    ```

    **Available arguments for `main.py`:**
    *   `--start_date`: Start date for data generation (YYYY-MM-DD, default: '2023-01-01').
    *   `--end_date`: End date for data generation (YYYY-MM-DD, default: '2023-01-31').
    *   `--frequency`: Frequency of data points (e.g., 'H' for hourly, 'D' for daily, default: 'H').
    *   `--window_size` (W): Size of the sliding window (default: 24).
    *   `--forecast_horizon` (H): Forecast horizon (default: 6).
    *   `--model_type`: Type of classifier ('RandomForest' or 'LogisticRegression', default: 'RandomForest').
    *   `--n_estimators`: Number of trees in RandomForest (only for RandomForest, default: 100).
    *   `--alert_threshold`: Probability threshold for triggering an alert (default: 0.5).
    *   `--random_state`: Random state for reproducibility (default: 42).

Upon successful execution, processed data, trained model, and evaluation results (metrics, plots) will be saved in the `data/` and `results/` directories, respectively.

## 5. Evaluation Metrics and Results
The `src/evaluate.py` script performs a comprehensive evaluation of the trained model. It loads the test data and the trained model, then calculates several key metrics to assess performance, particularly focusing on the positive class (incidents).

**Evaluation Metrics:**
*   **Accuracy**: Overall correctness of the model.
*   **Precision**: The proportion of positive identifications that were actually correct (True Positives / (True Positives + False Positives)). Crucial for minimizing false alarms.
*   **Recall**: The proportion of actual positives that were identified correctly (True Positives / (True Positives + False Negatives)). Important for not missing actual incidents.
*   **F1-Score**: The harmonic mean of Precision and Recall, providing a balance between the two.
*   **ROC AUC**: The Area Under the Receiver Operating Characteristic Curve. Measures the model's ability to distinguish between positive and negative classes across various thresholds.
*   **Confusion Matrix**: A table showing the number of true positives, true negatives, false positives, and false negatives.

**Alert Thresholding:**
An `alert_threshold` (default: 0.5) is applied to the predicted probabilities of an incident. If `P(incident) >= alert_threshold`, an alert is triggered. This allows for fine-tuning the balance between precision and recall based on operational requirements. Metrics like Alert Precision, Alert Recall, and Alert F1-Score are calculated based on this threshold.

**Outputs:**
*   **`results/roc_curve.png`**: A plot visualizing the ROC curve, illustrating the trade-off between the True Positive Rate and False Positive Rate.
*   **`results/evaluation_summary.txt`**: A text file containing a detailed summary of all calculated metrics, the confusion matrix, and a classification report.

## 6. Analysis of Results
Interpreting the evaluation results requires careful consideration:
*   **High Recall is often preferred for anomaly detection**: Missing an actual incident (False Negative) can be more costly than a false alarm (False Positive). However, excessively high false positives can lead to alert fatigue.
*   **ROC AUC**: A high ROC AUC (close to 1.0) indicates that the model is good at ranking positive samples higher than negative samples. It's a threshold-independent measure.
*   **Precision/Recall at specific threshold**: The alert threshold can be tuned to achieve a desired balance. For example, a higher threshold might increase precision (fewer false alarms) but decrease recall (potentially missing more incidents).
*   **Confusion Matrix**: Provides a direct view into the types of errors the model is making, which is invaluable for debugging and improvement.

The synthetic data setup is designed to generate a class imbalance (fewer incidents than normal operations), making metrics like F1-score and ROC AUC more informative than simple accuracy.

## 7. Limitations and Future Work
**Limitations:**
*   **Synthetic Data**: The current data is synthetically generated, which simplifies patterns. Real-world data can be far more complex, noisy, and have diverse anomaly types.
*   **Simple Models**: RandomForest and Logistic Regression are strong baselines but may not capture highly complex temporal dependencies or subtle anomalies as effectively as deep learning models (e.g., LSTMs, Transformers).
*   **Static Thresholds**: Incident labeling uses static thresholds. Adaptive or unsupervised anomaly detection methods might be more suitable for dynamic environments.
*   **Feature Engineering**: Currently, features are flattened raw window values. More sophisticated feature engineering (e.g., statistical features, domain-specific features) could improve performance.

**Future Work:**
*   **Advanced Models**: Experiment with deep learning models (RNNs, LSTMs, Transformers) or more advanced statistical models for time-series.
*   **Anomaly Detection Algorithms**: Explore unsupervised or semi-supervised anomaly detection algorithms (e.g., Isolation Forest, Autoencoders, One-Class SVM) that don't rely on labeled anomaly data.
*   **Real-World Data Integration**: Integrate with actual system monitoring data streams.
*   **Dynamic Thresholding**: Implement adaptive thresholds for incident detection and alerting.
*   **Interpretability**: Enhance model interpretability to understand *why* an anomaly was detected.

## 8. Real-World Adaptation
Adapting this project to real-world time-series anomaly detection systems would involve several key considerations:
*   **Data Sources**: Integrate with actual monitoring systems (e.g., Prometheus, Grafana, AWS CloudWatch, Azure Monitor) to collect real-time or near real-time metrics.
*   **Data Preprocessing**: Robust data cleaning, handling missing values, and normalization techniques will be crucial for noisy real-world data.
*   **Continuous Learning**: Implement a retraining pipeline to keep the model updated with evolving system behavior and new anomaly patterns.
*   **Alerting Systems**: Connect the `evaluate.py` output to actual alerting systems (e.g., Slack, PagerDuty, email) for operational teams.
*   **Scalability**: Ensure the data processing and model inference pipelines can scale to handle large volumes of time-series data.
*   **Feedback Loop**: Establish a feedback mechanism where operations teams can confirm or reject alerts, which can then be used to refine the model's performance and reduce false positives.
