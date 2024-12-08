import click
import os
import pandas as pd
import altair as alt
import aly

@click.command()
@click.option(
    '--train_path',
    default='data/processed/train_df.csv',
    type=click.Path(exists=True),
    help='Path to the input training CSV file.'
)
@click.option(
    '--test_path',
    default='data/processed/test_df.csv',
    type=click.Path(exists=True),
    help='Path to the input testing CSV file.'
)
@click.option(
    '--output_dir',
    default='results/figures/',
    type=click.Path(),
    help='Directory where output figures will be saved.'
)
def main(train_path, test_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    # EDA Steps:
    print(train_df.info())

    # Clean column names
    train_df.columns = train_df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    test_df.columns = test_df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Altair configurations
    alt.data_transformers.enable('default', max_rows=None)
    aly.alt.data_transformers.enable('vegafusion')

    # Univariate distribution for quantitative variables
    numeric_columns = [
        'age_(in_years)',
        'resting_blood_pressure_(in_mm_hg_on_admission_to_the_hospital)',
        'serum_cholesterol_(in_mg/dl)',
        'maximum_heart_rate_achieved',
        'st_depression_induced_by_exercise_relative_to_rest',
        'number_of_major_vessels_(0–3)_colored_by_fluoroscopy'
    ]

    # Visualization for numeric columns
    numeric_dist_plot = aly.dist(
        train_df[numeric_columns + ['diagnosis_of_heart_disease']],
        color='diagnosis_of_heart_disease'
    )
    numeric_dist_plot.save(f"{output_dir}/numeric_distributions.png")

    categorical_columns = [
        'sex', 
        'chest_pain_type', 
        'fasting_blood_sugar_>_120_mg/dl', 
        'resting_electrocardiographic_results',
        'exercise-induced_angina', 
        'slope_of_the_peak_exercise_st_segment', 
        'thalassemia'
    ]

    # Visualize categorical variables
    categorical_dist_plot = aly.dist(
        train_df[categorical_columns + ['diagnosis_of_heart_disease']]
        .assign(diagnosis_of_heart_disease=lambda x: x['diagnosis_of_heart_disease'].astype(object)), 
        dtype='object', 
        color='diagnosis_of_heart_disease'
    )
    categorical_dist_plot.save(f"{output_dir}/categorical_distributions.png")

    # Pairwise correlations for the numeric variables
    correlation_plot = aly.corr(train_df[numeric_columns])
    correlation_plot.save(f"{output_dir}/correlation_matrix.png")

    # Select numeric columns with at least one high correlation
    columns_with_at_least_one_high_corr = [
        'age_(in_years)',
        'resting_blood_pressure_(in_mm_hg_on_admission_to_the_hospital)',
        'serum_cholesterol_(in_mg/dl)',
        'maximum_heart_rate_achieved',
        'st_depression_induced_by_exercise_relative_to_rest',
        'number_of_major_vessels_(0–3)_colored_by_fluoroscopy',
        'diagnosis_of_heart_disease'
    ]

    sample_size = min(len(train_df), 300)

    pairwise_relationships_plot = aly.pair(
        train_df[columns_with_at_least_one_high_corr].sample(sample_size),
        color='diagnosis_of_heart_disease'
    )
    pairwise_relationships_plot.save(f"{output_dir}/pairwise_relationships.png")


if __name__ == "__main__":
    main()
