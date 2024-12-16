# 4_training_models.py
# author: Long Nguyen
# date: 2024-12-15
# Usage: python scripts/4_training_models.py --train data/processed/train_df.csv --seed 123 --write-to results

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
import warnings
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import click

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import parallel_backend
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (make_scorer, precision_score, recall_score, f1_score)
from src.class_model_trainer import class_model_trainer

# Suppress UndefinedMetricWarning when calculating precision for Dummy
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Suppress UserWarning when transforming test data
warnings.filterwarnings("ignore", category=UserWarning)


@click.command()
@click.option('--train', type=str, help="Location of train data file")
@click.option('--seed', type =int, help="Set seed for reproducibility")
@click.option('--write-to', type=str, help="Path to master directory where outputs will be written")

def main(train, seed, write_to):
    
    # Ensure necessary directories exist
    os.makedirs(os.path.join(write_to, "tables"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "models"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "figures"), exist_ok=True)

    print("Loading train data...")
    # Load train data
    train_data = pd.read_csv(train)

    # Split data into features and labels
    X_train, y_train = train_data.drop(columns='Diagnosis of heart disease'), train_data['Diagnosis of heart disease']

    # 1. DATA PREPROCESSOR
    categorical_features = [
        'Sex', 
        'Chest pain type', 
        'Fasting blood sugar > 120 mg/dl', 
        'Resting electrocardiographic results', 
        'Exercise-induced angina', 
        'Slope of the peak exercise ST segment', 
        'Thalassemia'
    ]
    numeric_features = list(set(X_train.columns) - set(categorical_features))
    
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", drop='if_binary', dtype=int, sparse_output=False),
    )
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    preprocessor = make_column_transformer(
        (categorical_transformer, categorical_features),
        (numeric_transformer, numeric_features),
    )

    # 2. CLASSIFICATION METRICS
    classification_metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label='> 50% diameter narrowing'),
        "recall": make_scorer(recall_score, pos_label='> 50% diameter narrowing'),
        "f1": make_scorer(f1_score, pos_label='> 50% diameter narrowing'),
    }
    
    print("Training models...")
    # 3. Training models
    models = class_model_trainer(preprocessor, X_train, y_train, pos_lable = '> 50% diameter narrowing', 
                        seed=seed, write_to=write_to, 
                        cv = 5, metrics = classification_metrics)

    print("Tuning model...")
    # 4. HYPERPARAMETER OPTIMIZATION
    param_distributions = {'logisticregression__C': np.logspace(-5, 5, 50)}
    custom_scorer = make_scorer(f1_score, pos_label='> 50% diameter narrowing')
    random_search = RandomizedSearchCV(
        models['logreg_bal'], 
        param_distributions=param_distributions,
        n_iter=100, n_jobs=-1, scoring=custom_scorer, random_state=123,
        return_train_score=True
    )
    
    with parallel_backend("multiprocessing"):
      with warnings.catch_warnings():
        random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
 
    # Save the best model
    with open(os.path.join(write_to, "models", "disease_pipeline.pickle"), 'wb') as f:
        pickle.dump(best_model, f)
    print("Best model saved.")
    
if __name__ == '__main__':
    main()
