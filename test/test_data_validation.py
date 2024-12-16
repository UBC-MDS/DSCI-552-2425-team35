# test_data_validation.py
# author: Sarah Eshafi
# date: 2024-12-14
# Attribution: This code was adapted from Tiffany Timbers

import pytest
import os
import numpy as np
import pandas as pd
import pandera as pa
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_validation import validate_data


# Test data setup
valid_data = pd.DataFrame({
    "Age (in years)": [63, 37, 41],
    "Sex": ["male", "female", "female"],
    "Chest pain type": ["typical angina", "non-anginal pain", "atypical angina"],
    "Resting blood pressure (in mm Hg on admission to the hospital)": [145, 130, 130],
    "Serum cholesterol (in mg/dl)": [233, 250, 204],
    "Fasting blood sugar > 120 mg/dl": [True, False, False],
    "Resting electrocardiographic results": ["normal", "having ST-T wave abnormality", "normal"],
    "Maximum heart rate achieved": [150, 187, 172],
    "Exercise-induced angina": ["no", "no", "yes"],
    "ST depression induced by exercise relative to rest": [2.3, 3.5, 1.4],
    "Slope of the peak exercise ST segment": ["downsloping", "flat", "upsloping"],
    "Number of major vessels (0–3) colored by fluoroscopy": [0.0, 0.0, 0.0],
    "Thalassemia": ["fixed defect", "reversible defect", "normal"],
    "Diagnosis of heart disease": ["< 50% diameter narrowing", "< 50% diameter narrowing", "> 50% diameter narrowing"]
})

# Case: wrong type passed to function
valid_data_as_np = valid_data.copy().to_numpy()
def test_valid_data_type():
    with pytest.raises(TypeError):
        validate_data(valid_data_as_np)

# Case: empty data frame
case_empty_data_frame = valid_data.copy().iloc[0:0]
def test_valid_data_empty_data_frame():
    with pytest.raises(ValueError):
        validate_data(case_empty_data_frame)

# Setup list of invalid data cases 
invalid_data_cases = []

# Case: label in "Diagnosis of heart disease" column encoded as 0 and 1, instead of proper labels
case_wrong_label_type = valid_data.copy()
case_wrong_label_type["Diagnosis of heart disease"] = case_wrong_label_type["Diagnosis of heart disease"].map({'< 50% diameter narrowing': 0, 
                                                                                                               '> 50% diameter narrowing': 1})
invalid_data_cases.append((case_wrong_label_type, "Check incorrect type for'Diagnosis of heart disease' values is missing or incorrect"))

# Case: wrong string value/category in "Diagnosis of heart disease" column
case_wrong_category_label = valid_data.copy()
case_wrong_category_label.loc[0, "Diagnosis of heart disease"] = "No narrowing"
invalid_data_cases.append((case_wrong_category_label, "Check absent or incorrect for wrong string value/category in 'Diagnosis of heart disease' column"))

# Case: missing value in "Diagnosis of heart disease" column
case_missing_class = valid_data.copy()
case_missing_class.loc[0, "Diagnosis of heart disease"] = None
invalid_data_cases.append((case_missing_class, "Check absent or incorrect for missing/null 'Diagnosis of heart disease' value"))

# Case: missing columns (one for each column)
columns = valid_data.columns
for col in columns:
    case_missing_col = valid_data.copy()
    case_missing_col = case_missing_col.drop(col, axis=1)  # drop column
    invalid_data_cases.append((case_missing_col, f"'{col}' is missing from DataFrameSchema"))

# Case: duplicate observations
case_duplicate = valid_data.copy()
case_duplicate = pd.concat([case_duplicate, case_duplicate.iloc[[0], :]], ignore_index=True)
invalid_data_cases.append((case_duplicate, f"Check absent or incorrect for duplicate rows"))

# Case: entire missing observation
case_missing_obs = valid_data.copy()
nan_row = pd.DataFrame([[np.nan] * (case_missing_obs.shape[1] - 1) + [np.nan]], columns=case_missing_obs.columns)
case_missing_obs = pd.concat([case_missing_obs, nan_row], ignore_index=True)
invalid_data_cases.append((case_missing_obs, f"Check absent or incorrect for missing observations (e.g., a row of all missing values)"))

# Parameterize invalid data test cases
@pytest.mark.parametrize("invalid_data, description", invalid_data_cases)
def test_valid_w_invalid_data(invalid_data, description):
    with pytest.raises(pa.errors.SchemaErrors) as excinfo:
        validate_data(invalid_data)