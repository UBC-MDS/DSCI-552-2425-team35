# 3_eda.py
# author: Hui Tang
# date: 2024-12-07

import click
import os
import pandas as pd
from src.eda_utils import (
    create_numeric_distributions,
    create_categorical_distributions,
    create_correlation_heatmap,
    save_high_correlations
)

@click.command()
@click.option(
    '--train',
    default='data/processed/train_df.csv',
    type=click.Path(exists=True),
    help='Path to the input training CSV file.'
)
@click.option(
    '--write-to',
    default='results',
    type=click.Path(),
    help='Directory where output figures will be saved.'
)
def main(train, write_to):
    # Ensure output directories exist
    os.makedirs(os.path.join(write_to, "figures"), exist_ok=True)
    os.makedirs(os.path.join(write_to, "tables"), exist_ok=True)

    # Load data
    train_df = pd.read_csv(train)

    # EDA Steps:
    numeric_columns = [
        'Age (in years)',
        'Resting blood pressure (in mm Hg on admission to the hospital)',
        'Serum cholesterol (in mg/dl)',
        'Maximum heart rate achieved',
        'ST depression induced by exercise relative to rest',
        'Number of major vessels (0–3) colored by fluoroscopy'
    ]

    categorical_columns = [
        'Sex',
        'Chest pain type',
        'Fasting blood sugar > 120 mg/dl',
        'Resting electrocardiographic results',
        'Exercise-induced angina',
        'Slope of the peak exercise ST segment',
        'Thalassemia'
    ]

    # Call EDA functions
    create_numeric_distributions(train_df, numeric_columns, os.path.join(write_to, "figures"))
    create_categorical_distributions(train_df, categorical_columns, os.path.join(write_to, "figures"))
    create_correlation_heatmap(train_df, numeric_columns, os.path.join(write_to, "figures"))
    save_high_correlations(train_df, numeric_columns, os.path.join(write_to, "tables"))


if __name__ == "__main__":
    main()
