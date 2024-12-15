# model_eval.py
# author: Marek Boulerice
# date: 2024-12-15

import os
import pandas as pd
from sklearn.metrics import f1_score, recall_score

def eval_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates a classification model given a predetermined set of evaluation metrics, and returns a data frame of the model's score

    This function evaluates the input model, which has been previously trained on the input train data.
    The function then scores the model based on the following predetermined criteria: F1 score, recall and accuracy. The model is scored on both input train and input test data, and the final results are compiled in a dataframe which is returned by the function.

    Parameters
    ----------
    model : sklearn.pipeline
        The classification model to be scored 

    X_train : pandas.DataFrame
        The DataFrame containing feature-data used to train the model
    
    y_train : pandas.DataFrame
        The DataFrame containing target-data used to train the model

    X_test : pandas.DataFrame
        The DataFrame containing feature-data used for final model evaluation
    
    y_test : pandas.DataFrame
        The DataFrame containing target-data used for final model evaluation

    Returns
    -------
    pandas.DataFrame
        A data frame containing model F1 score, recall and accuracy for train and test data
    """
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    metrics_df = pd.DataFrame({
        'Metric': ['F1 Score', 'Recall', 'Accuracy'],
        'Train': [
            f1_score(y_train, train_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            recall_score(y_train, train_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            model.score(X_train, y_train),
        ],
        'Test': [
            f1_score(y_test, test_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            recall_score(y_test, test_predictions, average='binary', pos_label='> 50% diameter narrowing'),
            model.score(X_test, y_test),
        ],
    })
    
    return metrics_df.round(3)


