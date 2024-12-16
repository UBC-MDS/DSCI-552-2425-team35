# test_model_eval.py
# author: Marek Boulerice
# date: 2024-12-15
# Attribution: This code was adapted from Tiffany Timbers

import pytest
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_eval import eval_model

# Test data setup

#Create dummy model

#create dummy X_train, y_train, X_test, y_test data
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



#Test cases to check

#1 passed an empty data frame

#2 Passed dataframe is wrong data type

#3 missing values in data frames

#4 failed to pass a model
