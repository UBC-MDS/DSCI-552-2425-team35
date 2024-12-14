# test_eda_utils.py
# author: Hui Tang
# date: 2024-12-14

import sys
import os
import pytest
import pandas as pd
import shutil

# Dynamically add the src directory to the Python path
# Adjust based on the actual structure in the image
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from eda_utils import (
    create_numeric_distributions,
    create_categorical_distributions,
    create_correlation_heatmap,
    save_high_correlations
)

def test_create_numeric_distributions():
    # Simulated input data for numeric distribution testing
    input_data = pd.DataFrame({
        'Age (in years)': [25, 35, 45, 55, 65],
        'Resting blood pressure (in mm Hg)': [120, 130, 125, 135, 140],
        'Diagnosis of heart disease': ['Yes', 'No', 'Yes', 'No', 'Yes']
    })

    output_dir = "test_figures_numeric"
    os.makedirs(output_dir, exist_ok=True)

    try:
        create_numeric_distributions(input_data, ['Age (in years)', 'Resting blood pressure (in mm Hg)'], output_dir)
        # Check if the output file is created
        output_path = os.path.join(output_dir, "numeric_distributions.png")
        assert os.path.exists(output_path), "Numeric distributions plot was not created."
    finally:
        shutil.rmtree(output_dir)


def test_create_categorical_distributions():
    # Simulated input data for categorical distribution testing
    input_data = pd.DataFrame({
        'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Chest pain type': ['Type1', 'Type2', 'Type3', 'Type2', 'Type1'],
        'Diagnosis of heart disease': ['Yes', 'No', 'Yes', 'No', 'Yes']
    })

    output_dir = "test_figures_categorical"
    os.makedirs(output_dir, exist_ok=True)

    try:
        create_categorical_distributions(input_data, ['Sex', 'Chest pain type'], output_dir)
        # Check if the output file is created
        output_path = os.path.join(output_dir, "categorical_distributions.png")
        assert os.path.exists(output_path), "Categorical distributions plot was not created."
    finally:
        shutil.rmtree(output_dir)


def test_create_correlation_heatmap():
    # Simulated input data for correlation heatmap testing
    input_data = pd.DataFrame({
        'Age (in years)': [25, 35, 45, 55, 65],
        'Resting blood pressure (in mm Hg)': [120, 130, 125, 135, 140],
        'Serum cholesterol (in mg/dl)': [200, 210, 220, 230, 240]
    })

    output_dir = "test_figures_heatmap"
    os.makedirs(output_dir, exist_ok=True)

    try:
        create_correlation_heatmap(input_data, ['Age (in years)', 'Resting blood pressure (in mm Hg)', 'Serum cholesterol (in mg/dl)'], output_dir)
        # Check if the output file is created
        output_path = os.path.join(output_dir, "correlation_matrix.png")
        assert os.path.exists(output_path), "Correlation heatmap was not created."
    finally:
        shutil.rmtree(output_dir)


def test_save_high_correlations():
    # Simulated input data for saving high correlations
    input_data = pd.DataFrame({
        'Age (in years)': [25, 35, 45, 55, 65],
        'Resting blood pressure (in mm Hg)': [120, 130, 125, 135, 140],
        'Serum cholesterol (in mg/dl)': [200, 210, 220, 230, 240]
    })

    output_dir = "test_tables"
    os.makedirs(output_dir, exist_ok=True)

    try:
        save_high_correlations(input_data, ['Age (in years)', 'Resting blood pressure (in mm Hg)', 'Serum cholesterol (in mg/dl)'], output_dir)
        # Check if the output files are created
        correlation_matrix_path = os.path.join(output_dir, "correlation_matrix.csv")
        high_correlation_path = os.path.join(output_dir, "high_correlations.csv")
        assert os.path.exists(correlation_matrix_path), "Correlation matrix file was not created."
        assert os.path.exists(high_correlation_path), "High correlations file was not created."
    finally:
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main()
