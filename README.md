# Heart Disease Predictor
Contributors: Sarah Eshafi, Hui Tang, Long Nguyen, Marek Boulerice

## About
This repository covers a machine learning model analysis with a goal to predict angiographic coronary disease in patients. Data is pulled from patients undergoing angiography at the Cleveland Clinic in Ohio. This analysis is composed of Exploratory Data Analysis, testing of various machine models on a training data set, model optimization via hyperparameter, and final model performance analysis. The final model is shown to have promising results, though limitations apply and further testing and optimization is recommended.

## Running the Report
To run the analysis for the first time, run the following from the root of this repository:

`conda-lock install --name breast-cancer-predictor conda-lock_[YOUR OS].yml`

Replace `[YOUR OS]` with the yml file name containing your operating system.

Then, run the following from the root of this repository:

`jupyter lab`

Upon opening jupyter lab (or your preferred Python IDE), open `heart_diagnostic_analysis.ipynb` and switch your kernel to "Python [conda env:522]".

Finally, click "Restart kernel and run all cells" to view the analysis.

## Dependencies
- conda (version 24.7.1 or higher)
- conda-lock (version 2.5.7 or higher)
- jupyterlab (version 4.2.5 or higher)
- nb_conda_kernels (version 2.5.1 or higher)
- Python and packages listed in environment.yml

## Licenses
The software code contained within this repository is licensed under the MIT license. See the [license file](https://github.com/UBC-MDS/DSCI-522-2425-team35-Heart_disease_diagnostic_machine/blob/main/LICENSE) for more information.