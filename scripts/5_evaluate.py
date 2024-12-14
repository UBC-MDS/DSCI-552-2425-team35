# 5_evaluate.py
# author: Hui Tang
# date: 2024-12-07

import pandas as pd
import os
import sys
import pickle
import click
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score

@click.command()
@click.option('--train', type=str, help="Location of train data file")
@click.option('--test', type=str, help="Path to the test data file", required=True)
@click.option('--pipeline', type=str, help="Path to the model pickle", required=True)
@click.option('--write-to', type=str, help="Path to the master directory where outputs will be written", required=True)
def main(train, test, pipeline, write_to):
    """
    Evaluate a trained model on test data and save evaluation metrics and confusion matrix.
    """
    # Define the path to the saved model file
    model_path = pipeline

    # Ensure necessary directories exist
    os.makedirs(os.path.join(write_to, "tables"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "figures"), exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist. Ensure it has been trained and saved.")

    # Load test data
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    # Split data into features and labels
    X_train, y_train = train_data.drop(columns='Diagnosis of heart disease'), train_data['Diagnosis of heart disease']
    X_test, y_test = test_data.drop(columns='Diagnosis of heart disease'), test_data['Diagnosis of heart disease']

    # Load the saved best model
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)

    # Evaluate the model
    train_predictions = best_model.predict(X_train)
    test_predictions = best_model.predict(X_test)
    metrics_df = pd.DataFrame({
        'Metric': ['F1 Score', 'Recall', 'Accuracy'],
        'Train': [
            f1_score(y_train, train_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            recall_score(y_train, train_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            best_model.score(X_train, y_train),
        ],
        'Test': [
            f1_score(y_test, test_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            recall_score(y_test, test_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            best_model.score(X_test, y_test),
        ],
    })
    metrics_df = metrics_df.round(3)
    metrics_df.to_csv(os.path.join(write_to, "tables", "model_metrics.csv"), index=False)

    # Save confusion matrix
    confmat = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, values_format="d")
    confmat.figure_.set_size_inches(10, 7)  # Set custom figure size
    confmat.figure_.tight_layout()
    confmat.figure_.savefig(
        os.path.join(write_to, "figures", "confusion_matrix.png"),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    main()
