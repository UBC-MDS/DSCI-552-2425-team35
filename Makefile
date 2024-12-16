# makefile
# author: Long Nguyen
# date: 2024-12-13

.PHONY: all clean

all: reports/heart_diagnostic_analysis.html reports/heart_diagnostic_analysis.pdf

# 1. Download and extract data
data/raw/pretransformed_heart_disease.csv: scripts/1_download_decode_data.py
	python scripts/1_download_decode_data.py \
		--id=45 \
		--write-to=data/raw

# 2. Read, validate, and split data
data/processed/train_df.csv data/processed/test_df.csv: scripts/2_data_split_validate.py \
data/raw/pretransformed_heart_disease.csv
	python scripts/2_data_split_validate.py \
		--split=0.1 \
		--raw-data=data/raw/pretransformed_heart_disease.csv \
		--write-to=data/processed

# 3. EDA
results/figures/numeric_distributions.png \
results/figures/categorical_distributions.png \
results/figures/correlation_matrix.png \
results/figures/pairwise_relationships.png: scripts/3_eda.py \
data/processed/train_df.csv
	python scripts/3_eda.py \
		--train data/processed/train_df.csv \
		--write-to results

# 4. Training models
results/tables/cross_val_std.csv results/tables/cross_val_score.csv results/models/disease_pipeline.pickle: scripts/4_training_models.py \
data/processed/train_df.csv
	python scripts/4_training_models.py \
			--train data/processed/train_df.csv \
			--seed 123 \
			--write-to results
		

# 5. Evaluate model
results/figures/confusion_matrix.png results/tables/model_metrics.csv: scripts/5_evaluate.py \
data/processed/train_df.csv \
data/processed/test_df.csv \
results/models/disease_pipeline.pickle
	python scripts/5_evaluate.py \
			--train data/processed/train_df.csv \
			--test data/processed/test_df.csv \
			--pipeline results/models/disease_pipeline.pickle \
			--write-to results



# Hey Marek! you only need to make changes from here down. The below is the template. Still looking for a command to automatically copy html to docs folder as index.html so we can render it to be landing page

# build HTML report and copy build to docs folder
reports/heart_diagnostic_analysis.html reports/heart_diagnostic_analysis.pdf : reports/heart_diagnostic_analysis.qmd \
reports/references.bib \
results/tables/model_metrics.csv \
data/processed/train_df.csv \
results/figures/categorical_distributions.png \
results/figures/numeric_distributions.png \
results/figures/correlation_matrix.png \
results/tables/cross_val_score.csv \
results/figures/confusion_matrix.png \
results/tables/model_metrics.csv
	quarto render report/adult_income_predictor_report.qmd --to html
	quarto render report/adult_income_predictor_report.qmd --to pdf

# clean up analysis / nuke everything
clean :
	rm -rf data/raw/*
	rm -rf data/logs/validation_errors.log \
			data/processed/cleaned_data.csv
	rm -rf results/figures/eda1.png \
			results/figures/eda2.png \
			results/figures/eda3.png \
			results/figures/eda4.png \
			results/figures/eda5.png \
			results/figures/eda6.png
	rm -rf data/processed/X_test.csv \
			data/processed/y_test.csv \
			results/models/model.pickle
	rm -rf results/figures/cm.png \
			results/table/test_score.csv
	rm -rf report/adult_income_predictor_report.html \
			report/adult_income_predictor_report.pdf \
			report/adult_income_predictor_report_files