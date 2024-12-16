# test_class_model_trainer.py
# author: Long Nguyen
# date: 2024-12-13

import os
import sys
import pytest
import pandas as pd
import shutil
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Dynamically add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from class_model_trainer import class_model_trainer


def test_class_model_trainer():
    # Simulated input data for testing
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(np.where(y == 1, '> 50% diameter narrowing', '<= 50% diameter narrowing'), name="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = [f"feature_{i}" for i in range(2)]
    numeric_features = [f"feature_{i}" for i in range(2, 10)]

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    preprocessor = make_column_transformer(
        (categorical_transformer, categorical_features),
        (numeric_transformer, numeric_features)
    )

    output_dir = "test_output"
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    try:
        models = class_model_trainer(
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            pos_lable='> 50% diameter narrowing',
            seed=123,
            write_to=output_dir,
            cv=3
        )

        # Assertions for models
        assert "dummy" in models, "Dummy model missing from output."
        assert "logreg" in models, "Logistic Regression model missing from output."
        assert "svc" in models, "Support Vector Classifier missing from output."

        # Assertions for output files
        std_csv = os.path.join(output_dir, "tables", "cross_val_std.csv")
        mean_csv = os.path.join(output_dir, "tables", "cross_val_score.csv")
        assert os.path.exists(std_csv), "Cross-validation std CSV not created."
        assert os.path.exists(mean_csv), "Cross-validation mean CSV not created."

        # Assertions for file content
        std_df = pd.read_csv(std_csv)
        mean_df = pd.read_csv(mean_csv)
        assert not std_df.empty, "Cross-validation std CSV is empty."
        assert not mean_df.empty, "Cross-validation mean CSV is empty."

    finally:
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main(["-v", "test/test_class_model_trainer.py"])
