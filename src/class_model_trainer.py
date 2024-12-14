# Class_model_trainer.py
# author: Long Nguyen
# date: 2024-12-13


# Core Libraries
import os  # For file path operations

# Data Manipulation
import pandas as pd  # For handling DataFrame operations

# Machine Learning
from sklearn.dummy import DummyClassifier  # For dummy classification model
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.svm import SVC  # For support vector classifier
from sklearn.pipeline import make_pipeline  # For creating pipelines
from sklearn.model_selection import cross_validate  # For cross-validation

# Metrics and Scoring
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score  # For metrics


def class_model_trainer(preprocessor, X_train, y_train, pos_lable, seed, write_to, cv = 5, metrics = None):
    """
    Train and evaluate multiple classification models using cross-validation.
    
    This function trains a variety of classification models, including a dummy classifier, 
    logistic regression, and support vector classifier (SVC), with optional class weight balancing. 
    It performs cross-validation using specified metrics and saves the results to CSV files.
    
    Parameters
    ----------
    preprocessor : sklearn.pipeline.Pipeline
        A preprocessing pipeline to transform the input data before training.
    X_train : pandas.DataFrame
        The training feature set.
    y_train : pandas.Series
        The training target variable.
    pos_lable : str
        The positive class label for evaluation metrics such as precision, recall, and F1-score.
    seed : int
        The random seed for reproducibility.
    write_to : str
        The directory path where the cross-validation results (CSV files) will be saved.
    cv : int, optional, default=5
        The number of cross-validation folds.
    metrics : dict, optional
        Custom scoring metrics for cross-validation. If not provided, the default is accuracy, 
        precision, recall, and F1-score.
    
    Returns
    -------
    dict
        A dictionary containing trained model pipelines:
        - "dummy": DummyClassifier pipeline.
        - "logreg": LogisticRegression pipeline.
        - "svc": Support Vector Classifier pipeline.
        - "logreg_bal": LogisticRegression pipeline with balanced class weights.
        - "svc_bal": Support Vector Classifier pipeline with balanced class weights.
    
    Output
    ------
    CSV files containing cross-validation results:
    - `cross_val_std.csv`: Standard deviation of cross-validation scores for each metric and model.
    - `cross_val_score.csv`: Mean cross-validation scores for each metric and model.
    
    Examples
    --------
    Define preprocessing steps and train models:
    
    >>> from sklearn.compose import make_column_transformer
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> 
    >>> categorical_transformer = make_pipeline(
    ...     SimpleImputer(strategy="most_frequent"),
    ...     OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ... )
    >>> numeric_transformer = make_pipeline(
    ...     SimpleImputer(strategy="median"),
    ...     StandardScaler()
    ... )
    >>> preprocessor = make_column_transformer(
    ...     (categorical_transformer, ["categorical_column"]),
    ...     (numeric_transformer, ["numerical_column"])
    ... )
    >>> 
    >>> models = class_model_trainer(
    ...     preprocessor=preprocessor,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     pos_lable="positive_class",
    ...     seed=42,
    ...     write_to="output_directory",
    ...     cv=3
    ... )
    """
    
    models = {
        "dummy": make_pipeline(DummyClassifier()),
        "logreg": make_pipeline(preprocessor, LogisticRegression(random_state=seed, max_iter=1000)),
        "svc": make_pipeline(preprocessor, SVC(random_state=seed)),
        "logreg_bal": make_pipeline(preprocessor, LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced")),
        "svc_bal": make_pipeline(preprocessor, SVC(random_state=seed, class_weight="balanced"))
    }

    metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=pos_lable),
        "recall": make_scorer(recall_score, pos_label=pos_lable),
        "f1": make_scorer(f1_score, pos_label=pos_lable),
    }
    
    cross_val_results = {}
    for model_name, pipeline in models.items():
        cross_val_results[model_name] = pd.DataFrame(
            cross_validate(pipeline, 
                           X_train, 
                           y_train, 
                           cv=cv, 
                           scoring=metrics, 
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

    return models