# data_validation.py
# author: Sarah Eshafi
# date: 2024-12-14

import pandas as pd
import pandera as pa


def validate_data(heart_df):
    """
    Validates the input cancer data in the form of a pandas DataFrame against a predefined schema,
    and returns the validated DataFrame.

    This function checks that the columns in the input DataFrame conform to the expected types and value ranges.
    It also ensures there are no duplicate rows and no entirely empty rows. 
    Finally, it checks the missingness threshold for most columns.

    Parameters
    ----------
    heart_df : pandas.DataFrame
        The DataFrame containing heart-related data, which includes columns such as 'age', 'sex', 
        'cholesterol', and other related measurements. The data is validated based on specific criteria for 
        each column.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame that conforms to the specified schema.

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame does not conform to the specified schema (e.g., incorrect data types, out-of-range values,
        duplicate rows, or empty rows).
    """
    if not isinstance(heart_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if heart_df.empty:
        raise ValueError("Dataframe must contain observations.")
    
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
            pa.Check(lambda heart_df: ~heart_df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda heart_df: ~(heart_df.isna().all(axis=1)).any(), error="Empty rows found.")
        ],
        drop_invalid_rows=True
    )
    
    return schema.validate(heart_df, lazy = True)