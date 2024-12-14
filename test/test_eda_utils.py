# test_eda_utils.py
# author: Hui Tang
# date: 2024-12-14

import pytest
import pandas as pd
import os
import shutil
from eda_utils import (
    create_numeric_distributions,
    create_categorical_distributions,
    create_correlation_heatmap,
    save_high_correlations
)

def test_create_numeric_distributions():
    # Test numeric distributions creation
    input_data = pd.DataFrame({
        "Age (in years)": [25, 35, 45, 55, 65],
        "Resting blood pressure (in mm Hg on admission to the hospital)": [120, 130, 125, 135, 140]
    })

    output_dir = "test_figures"
    os.makedirs(output_dir, exist_ok=True)

    try:
        create_numeric_distributions(input_data, input_data.columns, output_dir)
        # Check if output files are created
        assert len(os.listdir(output_dir)) > 0, "No distribution plots were created."
    finally:
        shutil.rmtree(output_dir)

def test_create_categorical_distributions():
    # Test categorical distributions creation
    input_data = pd.DataFrame({
        "Sex": ["Male", "Female", "Male", "Female", "Male"],
        "Chest pain type": ["Type1", "Type2", "Type3", "Type2", "Type1"]
    })

    output_dir = "test_figures"
    os.makedirs(output_dir, exist_ok=True)

    try:
        create_categorical_distributions(input_data, input_data.columns, output_dir)
        # Check if output files are created
        assert len(os.listdir(output_dir)) > 0, "No categorical plots were created."
    finally:
        shutil.rmtree(output_dir)

def test_create_correlation_heatmap():
    # Test correlation heatmap creation
    input_data = pd.DataFrame({
        "Age (in years)": [25, 35, 45, 55, 65],
        "Resting blood pressure (in mm Hg on admission to the hospital)": [120, 130, 125, 135, 140],
        "Serum cholesterol (in mg/dl)": [200, 210, 220, 230, 240]
    })

    output_dir = "test_figures"
    os.makedirs(output_dir, exist_ok=True)

    try:
        create_correlation_heatmap(input_data, input_data.columns, output_dir)
        # Check if output files are created
        assert len(os.listdir(output_dir)) > 0, "Correlation heatmap was not created."
    finally:
        shutil.rmtree(output_dir)

def test_save_high_correlations():
    # Test saving high correlations
    input_data = pd.DataFrame({
        "Age (in years)": [25, 35, 45, 55, 65],
        "Resting blood pressure (in mm Hg on admission to the hospital)": [120, 130, 125, 135, 140],
        "Serum cholesterol (in mg/dl)": [200, 210, 220, 230, 240]
    })

    output_dir = "test_tables"
    os.makedirs(output_dir, exist_ok=True)

    try:
        save_high_correlations(input_data, input_data.columns, output_dir)
        # Check if the table file is created
        assert len(os.listdir(output_dir)) > 0, "High correlations table was not created."
    finally:
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    pytest.main()
