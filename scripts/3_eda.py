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
    # Ensure output directory exists
    output_dir = os.path.join(write_to, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train)

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
    numeric_dist_plot.save(os.path.join(output_dir, "numeric_distributions.png"))

    # Visualization for categorical variables
    categorical_columns = [
        'Sex',
        'Chest pain type',
        'Fasting blood sugar > 120 mg/dl',
        'Resting electrocardiographic results',
        'Exercise-induced angina',
        'Slope of the peak exercise ST segment',
        'Thalassemia'
    ]

    categorical_dist_plot = aly.dist(
        train_df[categorical_columns + ['Diagnosis of heart disease']],
        dtype='object',
        color='Diagnosis of heart disease'
    )
    categorical_dist_plot.save(os.path.join(output_dir, "categorical_distributions.png"))

    # Pairwise correlations for numeric variables
    correlation_plot = aly.corr(train_df[numeric_columns])
    correlation_plot.save(os.path.join(output_dir, "correlation_matrix.png"))

    # Pairplot-like visualization (scatterplot matrix)
    pairwise_plot = alt.Chart(train_df.sample(min(len(train_df), 300))).mark_point().encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color='Diagnosis of heart disease'
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=numeric_columns[:3],  # Use the first 3 numeric columns for rows
        column=numeric_columns[:3]  # Use the first 3 numeric columns for columns
    )
    pairwise_plot.save(os.path.join(output_dir, "pairwise_relationships.png"))


if __name__ == "__main__":
    main()
