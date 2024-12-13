

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