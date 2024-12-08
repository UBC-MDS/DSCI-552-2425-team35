# 4_training_models.py
# author: Long Nguyen
# date: 2024-12-07

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import click
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


@click.command()
@click.option('--train', type=str, help="Location of train data file")
@click.option('--write-to', type=str, help="Path to master directory where outputs will be written")
def main(train, write_to):
    # Ensure necessary directories exist
    os.makedirs(os.path.join(write_to, "tables"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "models"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "figures"), exist_ok=True)

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
    
    # 3. TRAINING MODELS
    models = {
        "dummy": make_pipeline(DummyClassifier()),
        "logreg": make_pipeline(preprocessor, LogisticRegression(random_state=123, max_iter=1000)),
        "svc": make_pipeline(preprocessor, SVC(random_state=123)),
        "logreg_bal": make_pipeline(preprocessor, LogisticRegression(random_state=123, max_iter=1000, class_weight="balanced")),
        "svc_bal": make_pipeline(preprocessor, SVC(random_state=123, class_weight="balanced"))
    }
    
    cross_val_results = {}
    for model_name, pipeline in models.items():
        cross_val_results[model_name] = pd.DataFrame(
            cross_validate(pipeline, 
                           X_train, 
                           y_train, 
                           cv=5, 
                           scoring=classification_metrics, 
                           return_train_score=True)
        ).agg(['mean', 'std']).round(3).T
    
    # Save cross-validation results (standard deviation)
    std_df = pd.concat(cross_val_results, axis='columns').reset_index()
    std_df.to_csv(
        os.path.join(write_to, "tables", "cross_val_std.csv"), index=False
    )
    
    # Save cross-validation results (mean)
    mean_df = pd.concat(cross_val_results, axis='columns').reset_index()
    mean_df.to_csv(
        os.path.join(write_to, "tables", "cross_val_score.csv"), index=False
    )
    
    # 4. HYPERPARAMETER OPTIMIZATION
    param_distributions = {'logisticregression__C': np.logspace(-5, 5, 50)}
    custom_scorer = make_scorer(f1_score, pos_label='> 50% diameter narrowing')
    random_search = RandomizedSearchCV(
        models['logreg_bal'], 
        param_distributions=param_distributions,
        n_iter=100, n_jobs=-1, scoring=custom_scorer, random_state=123,
        return_train_score=True
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Save the best model
    with open(os.path.join(write_to, "models", "disease_pipeline.pickle"), 'wb') as f:
        pickle.dump(best_model, f)

if __name__ == '__main__':
    main()
