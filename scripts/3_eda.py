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
    os.makedirs(os.path.join(write_to, "figures"), exist_ok=True)

    # Load data
    train_df = pd.read_csv(train)

    # EDA Steps:
    print(train_df.info())

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

    categorical_columns = [
        'Sex', 
        'Chest pain type', 
        'Fasting blood sugar > 120 mg/dl', 
        'Resting electrocardiographic results',
        'Exercise-induced angina', 
        'Slope of the peak exercise ST segment', 
        'Thalassemia'
    ]

    # Visualize categorical variables
    categorical_dist_plot = aly.dist(
        train_df[categorical_columns + ['Diagnosis of heart disease']]
        .assign(diagnosis_of_heart_disease=lambda x: x['Diagnosis of heart disease'].astype(object)), 
        dtype='object', 
        color='Diagnosis of heart disease'
    )
    categorical_dist_plot.save(os.path.join(write_to, "figures", "categorical_distributions.png"))

    # Pairwise correlations for the numeric variables
    correlation_plot = aly.corr(train_df[numeric_columns])
    correlation_plot.save(os.path.join(write_to, "figures", "correlation_matrix.png"))

    # Select numeric columns with at least one high correlation
    columns_with_at_least_one_high_corr = [
        'Age (in years)',
        'Resting blood pressure (in mm Hg on admission to the hospital)',
        'Serum cholesterol (in mg/dl)',
        'Maximum heart rate achieved',
        'ST depression induced by exercise relative to rest',
        'Number of major vessels (0–3) colored by fluoroscopy',
        'Diagnosis of heart disease'
    ]

    sample_size = min(len(train_df), 300)

    # Pairplot-like visualization
    pairwise_plot = alt.Chart(train_df).mark_point().encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color='Diagnosis of heart disease'
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=['Age', 'Resting BP', 'Cholesterol'],
        column=['Age', 'Resting BP', 'Cholesterol']
    )
    
    pairwise_plot.save('pairwise_relationships.png')


if __name__ == "__main__":
    main()
