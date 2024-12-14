# 3_eda.py
# author: Hui Tang
# date: 2024-12-07

import click
import os
import pandas as pd
import altair as alt
import altair_ally as aly

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
    # Altair configurations
    alt.data_transformers.enable('default', max_rows=None)
    aly.alt.data_transformers.enable('vegafusion')

    # Univariate distribution for quantitative variables
    numeric_columns = [
        'Age (in years)',
        'Resting blood pressure (in mm Hg on admission to the hospital)',
        'Serum cholesterol (in mg/dl)',
        'Maximum heart rate achieved',
        'ST depression induced by exercise relative to rest',
        'Number of major vessels (0–3) colored by fluoroscopy'
    ]

    # Visualization for numeric columns
    numeric_dist_plot = aly.dist(
        train_df[numeric_columns + ['Diagnosis of heart disease']],
        color='Diagnosis of heart disease'
    )
    numeric_dist_plot.save(os.path.join(write_to, "figures", "numeric_distributions.png"))

    # Remove nulls from relevant categorical columns, especially "Thalassemia"
    categorical_columns = [
        'Sex',
        'Chest pain type',
        'Fasting blood sugar > 120 mg/dl',
        'Resting electrocardiographic results',
        'Exercise-induced angina',
        'Slope of the peak exercise ST segment',
        'Thalassemia'
    ]
    train_df = train_df.dropna(subset=categorical_columns)

    # Visualize categorical variables
    categorical_dist_plot = aly.dist(
        train_df[categorical_columns + ['Diagnosis of heart disease']]
        .assign(diagnosis_of_heart_disease=lambda x: x['Diagnosis of heart disease'].astype(object)),
        dtype='object',
        color='Diagnosis of heart disease'
    )
    categorical_dist_plot.save(os.path.join(write_to, "figures", "categorical_distributions.png"))

    # Pairwise correlations for the numeric variables
    correlation_matrix = train_df[numeric_columns].corr()
    correlation_matrix.to_csv(os.path.join(write_to, "tables", "correlation_matrix.csv"))

    # Identify high correlations
    high_corr = correlation_matrix.stack().reset_index()
    high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
    high_corr = high_corr[
        (high_corr['Variable 1'] != high_corr['Variable 2']) &
        (high_corr['Correlation'].abs() > 0.7)  # Threshold for "high correlation"
    ].sort_values(by='Correlation', ascending=False)

    high_corr.to_csv(os.path.join(write_to, "tables", "high_correlations.csv"), index=False)

    # Print the highly correlated variables to the console (optional)
    print("Highly correlated variables (|correlation| > 0.7):")
    print(high_corr)

    # Save correlation heatmap
    correlation_plot = aly.corr(train_df[numeric_columns])
    correlation_plot.save(os.path.join(write_to, "figures", "correlation_matrix.png"))

    # Pairplot-like visualization
    sample_size = min(len(train_df), 300)
    pairwise_plot = alt.Chart(train_df.sample(sample_size)).mark_point().encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color='Diagnosis of heart disease'
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=numeric_columns[:3],  # Use first three numeric columns for demonstration
        column=numeric_columns[:3]
    )
    pairwise_plot.save(os.path.join(write_to, "figures", "pairwise_relationships.png"))


if __name__ == "__main__":
    main()
