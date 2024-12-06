# 2_data_split_validate.py
# author: Sarah Eshafi
# date: 2024-12-05

import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import pandera as pa
from deepchecks.tabular.checks import FeatureDrift
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation
from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.checks.data_integrity import FeatureFeatureCorrelation
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--split', type=float, help="Proportion of data to use as test data")
@click.option('--raw-data', type=str, help="Location of pre-processed data file")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")

def main(split, raw_data, write_to):
    """Validates data and exports two csv files as train test split."""
    # fetch dataset
    df = pd.read_csv(raw_data)

    # Initial data cleaning
    df = df[df['Diagnosis of heart disease'] <= 1]
    df['Diagnosis of heart disease'] = df['Diagnosis of heart disease'].replace(
        {0: '< 50% diameter narrowing', 1: '> 50% diameter narrowing'})
    
        #Validation data step: 
    # - check correct column types, 
    # - check missingess threshold for most columns, 
    # - check correct expected values for categorical columns
    # - remove unexpected values
    
    schema = pa.DataFrameSchema(
        {
            "Age (in years)": pa.Column(int,
                                        pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                            element_wise=False,
                                            error="Too many null values in column."), 
                                        nullable=True),
            "Sex": pa.Column(str,pa.Check.isin(["male", "female"])),
            "Chest pain type": pa.Column(str, pa.Check.isin(["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])),
            "Resting blood pressure (in mm Hg on admission to the hospital)": pa.Column(int,
                                                                                pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                                                                    element_wise=False, 
                                                                                    error="Too many null values in column."), 
                                                                                nullable=True),
            "Serum cholesterol (in mg/dl)": pa.Column(int,
                                                pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                                    element_wise=False, 
                                                    error="Too many null values in column."), 
                                                nullable=True),
            "Fasting blood sugar > 120 mg/dl": pa.Column(bool,
                                                    pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                                        element_wise=False, 
                                                        error="Too many null values in column."), 
                                                    nullable=True),
            "Resting electrocardiographic results": pa.Column(str,pa.Check.isin(["normal", 
                                                                                 "having ST-T wave abnormality", 
                                                                                 "showing probable or definite left ventricular hypertrophy by Estes' criteria"])),
            "Maximum heart rate achieved": pa.Column(int,
                                                pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                                    element_wise=False, 
                                                    error="Too many null values in column."), 
                                                nullable=True),
            "Exercise-induced angina": pa.Column(str, pa.Check.isin(["yes","no"])),
            "ST depression induced by exercise relative to rest": pa.Column(float,
                                                                    pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                                                        element_wise=False, 
                                                                        error="Too many null values in column."), 
                                                                    nullable=True),
            "Slope of the peak exercise ST segment": pa.Column(str, pa.Check.isin(["upsloping", "flat", "downsloping"])),
            "Number of major vessels (0–3) colored by fluoroscopy": pa.Column(float,
                                                                        pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                                                        element_wise=False, 
                                                                        error="Too many null values in column."),
                                                                    nullable=True),
            "Thalassemia": pa.Column(str, 
                                     pa.Check(lambda s: s.isna().mean() <= 0.05, 
                                            element_wise=False, 
                                            error="Too many null values in column."),
                                        nullable=True),
            "Diagnosis of heart disease": pa.Column(str, pa.Check.isin(["< 50% diameter narrowing", "> 50% diameter narrowing"]))
        } ,
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ],
        drop_invalid_rows=True
    )
    
    df = schema.validate(df, lazy = True)

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=split)
    
    # X_train = train_df.drop('Diagnosis of heart disease', axis=1)
    # y_train = train_df['Diagnosis of heart disease']
    # X_test = test_df.drop('Diagnosis of heart disease', axis=1)
    # y_test = test_df['Diagnosis of heart disease']

    #verify correlations - Feature-target:
    train_ds = Dataset(train_df, label="Diagnosis of heart disease", cat_features=['Sex','Chest pain type',
                                                                                   'Fasting blood sugar > 120 mg/dl',
                                                                                   'Resting electrocardiographic results',
                                                                                   'Exercise-induced angina',
                                                                                   'Slope of the peak exercise ST segment',
                                                                                   'Thalassemia'])
    test_ds = Dataset(test_df, label="Diagnosis of heart disease", cat_features=['Sex','Chest pain type',
                                                                               'Fasting blood sugar > 120 mg/dl',
                                                                               'Resting electrocardiographic results',
                                                                               'Exercise-induced angina',
                                                                               'Slope of the peak exercise ST segment',
                                                                               'Thalassemia'])
    
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=train_ds)

    #verify correlations: feature-feature:
    from deepchecks.tabular.datasets.classification import adult
    from deepchecks.tabular.checks.data_integrity import FeatureFeatureCorrelation
    
    check = FeatureFeatureCorrelation()
    
    check.add_condition_max_number_of_pairs_above_threshold(0.8, 3)
    result = check.run(train_ds)

    # Verify data drift
    check = FeatureDrift()
    result = check.run(train_dataset=train_ds, test_dataset=test_ds)    

    # Save the DataFrame to a CSV file
    train_df.to_csv(os.path.join(write_to, "train_df.csv"), index=False)
    test_df.to_csv(os.path.join(write_to, "test_df.csv"), index=False)

if __name__ == '__main__':
    main()