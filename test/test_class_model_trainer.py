# test_class_model_trainer.py
# author: Long Nguyen
# date: 2024-12-13

import os
import sys

# Add the parent directory of `src` to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.exceptions import UndefinedMetricWarning

# Suppress UndefinedMetricWarning when calculating precision for Dummy
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Import the function to test
from src.class_model_trainer import class_model_trainer

def test_class_model_trainer():
    """
    Test script for validating the functionality of the `class_model_trainer` function.
    
    This script:
    1. Creates a synthetic classification dataset using `make_classification`.
    2. Prepares the dataset for training by applying preprocessing pipelines for categorical and numerical features.
    3. Splits the dataset into training and test sets.
    4. Trains a series of models using the `class_model_trainer` function and writes the results to specified output files.
    5. Asserts that the expected models are trained and the output files (cross-validation scores) are generated.
    6. Validates that the output files contain data.
    
    Modules and Libraries:
    - `os`, `sys`: For managing paths and directories.
    - `pandas`, `numpy`: For data manipulation.
    - `warnings`: To suppress specific warnings during testing.
    - `sklearn`: For dataset generation, model training, and preprocessing pipelines.
    
    Key Steps:
    - Preprocessing: Applies `SimpleImputer`, `OneHotEncoder`, and `StandardScaler` to handle missing values, encode categorical variables, and scale numeric features.
    - Model Training: Utilizes `class_model_trainer` to train models and performs 3-fold cross-validation.
    - Output Validation: Checks if expected models are returned and corresponding CSV files are created and populated.
    
    Expected Output:
    - Model dictionary containing trained models (`dummy`, `logreg`, `svc`).
    - Two CSV files (`cross_val_std.csv` and `cross_val_score.csv`) saved under the `test_output/tables` directory.
    
    Usage:
    - Run the script directly to execute the test.
    - Ensure the `src.class_model_trainer` module is accessible from the `src` directory.
    
    Raises:
    - AssertionError: If any of the validation steps fail.
    
    Example:
        $ python test_class_model_trainer.py
    """
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=100, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2, 
        random_state=42
    )
    # Convert to DataFrame for compatibility
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(np.where(y == 1, '> 50% diameter narrowing', '<= 50% diameter narrowing'), name="target")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    categorical_features = [f"feature_{i}" for i in range(2)]  # Assume first 2 features are categorical
    numeric_features = [col for col in X.columns if col not in categorical_features]

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

    # Define paths for output
    write_to = "test_output"
    os.makedirs(os.path.join(write_to, "tables"), exist_ok=True)

    # Train models and capture outputs
    models = class_model_trainer(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        pos_lable='> 50% diameter narrowing',
        seed=123,
        write_to=write_to,
        cv=3
    )

    # Assertions to verify functionality
    assert "dummy" in models, "Dummy model missing from output."
    assert "logreg" in models, "Logistic Regression model missing from output."
    assert "svc" in models, "Support Vector Classifier missing from output."
    assert os.path.exists(os.path.join(write_to, "tables", "cross_val_std.csv")), "Cross-validation std CSV not created."
    assert os.path.exists(os.path.join(write_to, "tables", "cross_val_score.csv")), "Cross-validation mean CSV not created."

    # Load and check saved CSVs
    std_df = pd.read_csv(os.path.join(write_to, "tables", "cross_val_std.csv"))
    mean_df = pd.read_csv(os.path.join(write_to, "tables", "cross_val_score.csv"))
    assert not std_df.empty, "Cross-validation std CSV is empty."
    assert not mean_df.empty, "Cross-validation mean CSV is empty."

    print("All tests passed.")

# Run the test function
if __name__ == "__main__":
    test_class_model_trainer()
