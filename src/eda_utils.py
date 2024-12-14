# eda_utils.py
# author: Hui Tang
# date: 2024-12-13

import os
import pandas as pd
import altair as alt
import altair_ally as aly

def create_numeric_distributions(train_df, numeric_columns, output_dir):
    """Creates and saves numeric distribution plots."""
    numeric_dist_plot = aly.dist(
        train_df[numeric_columns + ['Diagnosis of heart disease']],
        color='Diagnosis of heart disease'
    )
    output_path = os.path.join(output_dir, "numeric_distributions.png")
    numeric_dist_plot.save(output_path)
    print(f"Numeric distributions saved to {output_path}")


def create_categorical_distributions(train_df, categorical_columns, output_dir):
    """Creates and saves categorical distribution plots."""
    train_df = train_df.dropna(subset=categorical_columns)  # Remove nulls
    categorical_dist_plot = aly.dist(
        train_df[categorical_columns + ['Diagnosis of heart disease']]
        .assign(diagnosis_of_heart_disease=lambda x: x['Diagnosis of heart disease'].astype(object)),
        dtype='object',
        color='Diagnosis of heart disease'
    )
    output_path = os.path.join(output_dir, "categorical_distributions.png")
    categorical_dist_plot.save(output_path)
    print(f"Categorical distributions saved to {output_path}")


def create_correlation_heatmap(train_df, numeric_columns, output_dir):
    """Creates and saves the correlation heatmap."""
    correlation_plot = aly.corr(train_df[numeric_columns])
    output_path = os.path.join(output_dir, "correlation_matrix.png")
    correlation_plot.save(output_path)
    print(f"Correlation heatmap saved to {output_path}")


def save_high_correlations(train_df, numeric_columns, output_dir):
    """Identifies and saves high correlations."""
    correlation_matrix = train_df[numeric_columns].corr()
    correlation_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

    high_corr = correlation_matrix.stack().reset_index()
    high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
    high_corr = high_corr[
        (high_corr['Variable 1'] != high_corr['Variable 2']) &
        (high_corr['Correlation'].abs() > 0.7)  # Threshold for "high correlation"
    ].sort_values(by='Correlation', ascending=False)

    high_corr.to_csv(os.path.join(output_dir, "high_correlations.csv"), index=False)
    print("High correlations saved to high_correlations.csv")
