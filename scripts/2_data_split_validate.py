# 2_data_split_validate.py
# author: Sarah Eshafi
# date: 2024-12-05

import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from deepchecks.tabular.checks import FeatureDrift
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation
from deepchecks.tabular.datasets.classification import adult
from deepchecks.tabular.checks.data_integrity import FeatureFeatureCorrelation
from sklearn.model_selection import train_test_split
from src.data_validation import validate_data

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

    print(df.head())
    
    # Validate data using function 
    df = validate_data(df)
    print(df.head())

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=split)

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
