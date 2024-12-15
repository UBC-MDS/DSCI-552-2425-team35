# Heart Disease Predictor
Contributors: Sarah Eshafi, Hui Tang, Long Nguyen, Marek Boulerice

## About
This repository covers the creation of a machine learning model analysis with a goal to predict angiographic coronary disease in patients. Data is pulled from patients undergoing angiography at the Cleveland Clinic in Ohio. This analysis is composed of Exploratory Data Analysis, testing of various machine models on a training data set, model optimization via hyperparameter, and final model performance analysis. The final LogisticRegression model is shown to perform quite well on testing data, reporting final F1 scores of around 0.89. The Recall value on this model is also quite promising, at around 0.9. This indicates the model is able to predict the occurrence of heart disease consistently with very few false negative cases (misses). Nevertheless, a few limitations still apply to our model. while the recall is quite low, even missing a few cases of heart disease in patients would have devastating effects on lives of patients. As well, the model requires quite extensive medical information from patients to deliver an accurate prediction, which may make it difficult to roll out in commercial applications. With these factors in mind we suggest further tuning of the model, though initial results are promising.


The final model is shown to have promising results, though limitations apply and further testing and optimization is recommended.

## Running the Report
To run the analysis:
#### 1\. Using Docker

*note - the instructions in this section also depends on running this in
a unix shell (e.g., terminal or Git Bash)*

To replicate the analysis, install
[Docker](https://www.docker.com/get-started). Then clone this GitHub
repository and run the following command at the command line/terminal
from the root directory of this project:

    docker compose up

Copy the link from the output (the link would look like below)
![Jupyter-lab](img/jl-link.png)

and paste it to your browser and change the port number from `8888` to `9999` to launch jupyter notebook.
![Jupyter-lab](img/9999.png)

#### 2\. Running the Analysis
To run the analysis, open a terminal and run the following commands:
```
python scripts/1_download_decode_data.py --id=45 --write-to=data/raw

python scripts/2_data_split_validate.py --split=0.2 --raw-data=data/raw/pretransformed_heart_disease.csv --write-to=data/processed

python scripts/3_eda.py  --train data/processed/train_df.csv --write-to results

python scripts/4_training_models.py --train data/processed/train_df.csv --seed 123 --write-to results

python scripts/5_evaluate.py --train data/processed/train_df.csv \
                                --test data/processed/test_df.csv \
                                --pipeline results/models/disease_pipeline.pickle \
                                --write-to results

quarto render reports/heart_diagnostic_analysis.qmd --to html
```

#### 3\. Clean Up
To shut down the container and clean up the resources, type Cntrl + C in the terminal where you launched the container, and then type `docker compose rm`.

## Dependencies
- conda (version 24.7.1 or higher)
- conda-lock (version 2.5.7 or higher)
- jupyterlab (version 4.2.5 or higher)
- nb_conda_kernels (version 2.5.1 or higher)
- Python and packages listed in environment.yml

## Licenses
The software code contained within this repository is licensed under the MIT license. See the [license file](https://github.com/UBC-MDS/DSCI-522-2425-team35-Heart_disease_diagnostic_machine/blob/main/LICENSE) for more information.
