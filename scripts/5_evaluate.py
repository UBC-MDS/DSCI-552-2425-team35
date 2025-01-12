# 5_evaluate.py
# author: Hui Tang
# date: 2024-12-07
# Usage: python scripts/5_evaluate.py --train data/processed/train_df.csv \
                                # --test data/processed/test_df.csv \
                                # --pipeline results/models/disease_pipeline.pickle \
                                # --write-to results

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import pickle
import click
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score
from src.model_eval import eval_model

@click.command()
@click.option('--train', type=str, help="Location of train data file")
@click.option('--test', type=str, help="Path to the test data file", required=True)
@click.option('--pipeline', type=str, help="Path to the model pickle", required=True)
@click.option('--write-to', type=str, help="Path to the master directory where outputs will be written", required=True)
def main(train, test, pipeline, write_to):
    """
    Evaluate a trained model on test data and save evaluation metrics and confusion matrix.

    Usage: 
    python scripts/5_evaluate.py --train data/processed/train_df.csv \
                                --test data/processed/test_df.csv \
                                --pipeline results/models/disease_pipeline.pickle \
                                --write-to results
    
    """

    # Check if the model file exists
    if not os.path.exists(pipeline):
        raise FileNotFoundError(f"The model file {pipeline} does not exist. Ensure it has been trained and saved.")

    # Load train and test data
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    # Split data into features and labels
    X_train, y_train = train_data.drop(columns='Diagnosis of heart disease'), train_data[['Diagnosis of heart disease']]
    X_test, y_test = test_data.drop(columns='Diagnosis of heart disease'), test_data[['Diagnosis of heart disease']]

    # Load the saved best model
    print(f"Loading model from: {pipeline}")
    with open(pipeline, 'rb') as f:
        best_model = pickle.load(f)
    print(f"Model loaded successfully.")

    # Evaluate the model
    metrics_df = eval_model(best_model,X_train, y_train, X_test, y_test)

    #Save model score to csv
    metrics_df.to_csv(os.path.join(write_to, "tables", "model_metrics.csv"), index=False)

    # Save confusion matrix
    confmat = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, values_format="d")
    confmat.figure_.set_size_inches(10, 7)  # Set custom figure size
    confmat.figure_.tight_layout()
    confmat.figure_.savefig(
        os.path.join(write_to, "figures", "confusion_matrix.png"),
        bbox_inches='tight'
    )
    print("Evaluation complete. Results saved to:", write_to)


if __name__ == '__main__':
    main()
