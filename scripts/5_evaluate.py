# 5_evaluate.py
# author: Hui Tang
# date: 2024-12-07

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score, accuracy_score
import click

@click.command()
@click.option('--test', type=str, help="Path to the test data file", required=True)
@click.option('--write-to', type=str, help="Path to the master directory where outputs will be written", required=True)
def main(test, write_to):
    """
    Evaluate a trained model on test data and save evaluation metrics and confusion matrix.
    """
    # Define the path to the saved model file
    model_path = os.path.join(write_to, "models", "disease_pipeline.pickle")

    # Ensure necessary directories exist
    os.makedirs(os.path.join(write_to, "tables"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "figures"), exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist. Ensure it has been trained and saved.")

    # Load test data
    test_data = pd.read_csv(test)
    if 'target' not in test_data.columns:
        raise ValueError("The test dataset must contain a 'target' column.")
    X_test, y_test = test_data.drop(columns='target'), test_data['target']

    # Load the saved best model
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)

    # Evaluate the model
    test_predictions = best_model.predict(X_test)
    metrics_df = pd.DataFrame({
        'Metric': ['F1 Score', 'Recall', 'Accuracy'],
        'Test': [
            f1_score(y_test, test_predictions, pos_label='> 50% diameter narrowing'),
            recall_score(y_test, test_predictions, pos_label='> 50% diameter narrowing'),
            accuracy_score(y_test, test_predictions),
        ],
    })
    metrics_df.to_csv(os.path.join(write_to, "tables", "evaluation_metrics.csv"), index=False)

    # Save confusion matrix
    confmat = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, values_format="d")
    plt.savefig(os.path.join(write_to, "figures", "evaluation_confusion_matrix.png"))
    plt.close()

if __name__ == '__main__':
    main()
