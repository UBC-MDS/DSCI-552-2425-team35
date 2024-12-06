# 1_download_decode_data.py
# author: Sarah Eshafi
# date: 2024-12-05

import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ucimlrepo import fetch_ucirepo
import pandas as pd

@click.command()
@click.option('--id', type=int, help="ID of the UCI repo dataset to download")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")

def main(id, write_to):
    """Downloads data from the UCI package to a local filepath and decodes variables and column headers."""
    # fetch dataset
    heart_disease = fetch_ucirepo(id=id) 
    data = heart_disease.data

    # Extract the features, targets, and headers
    features = data['features']  # The feature columns
    targets = data['targets']    # The target column
    headers = data['headers']    # The headers, if needed
    
    # Merge features and targets into a single DataFrame
    heart_disease_df = pd.concat([features, targets], axis=1)
    
    # Set the column names from headers
    heart_disease_df.columns = headers

    # Decode the values
    heart_disease_df['sex'] = heart_disease_df['sex'].replace({1: 'male', 0: 'female'})
    heart_disease_df['cp'] = heart_disease_df['cp'].replace({1: 'typical angina', 2: 'atypical angina', 
                                                             3:'non-anginal pain', 4:'asymptomatic'})
    heart_disease_df['restecg'] = heart_disease_df['restecg'].replace({0: 'normal', 1: 'having ST-T wave abnormality', 
                                                                       2:"showing probable or definite left ventricular hypertrophy by Estes' criteria"})
    heart_disease_df['fbs'] = heart_disease_df['fbs'].replace({0: 'False', 1: 'True'})
    heart_disease_df['exang'] = heart_disease_df['exang'].replace({0: 'no', 1: 'yes'})
    heart_disease_df['exang'] = heart_disease_df['exang'].replace({0: 'no', 1: 'yes'})
    heart_disease_df['slope'] = heart_disease_df['slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'})
    heart_disease_df['thal'] = heart_disease_df['thal'].replace({3: 'normal', 6: 'fixed defect', 7: 'reversable defect'})

    # Set the feature names
    feature_names = [
        "Age (in years)",
        "Sex",
        "Chest pain type",
        "Resting blood pressure (in mm Hg on admission to the hospital)",
        "Serum cholesterol (in mg/dl)",
        "Fasting blood sugar > 120 mg/dl",
        "Resting electrocardiographic results",
        "Maximum heart rate achieved",
        "Exercise-induced angina",
        "ST depression induced by exercise relative to rest",
        "Slope of the peak exercise ST segment",
        "Number of major vessels (0–3) colored by fluoroscopy",
        "Thalassemia",
        "Diagnosis of heart disease"
    ]
    
    heart_disease_df.columns = feature_names

    # Save the DataFrame to a CSV file
    heart_disease_df.to_csv(os.path.join(write_to, "pretransformed_heart_disease.csv"), index=False)

if __name__ == '__main__':
    main()