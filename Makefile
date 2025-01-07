# makefile
# author: Long Nguyen
# date: 2024-12-13

.PHONY: all clean

all: reports/heart_diagnostic_analysis.html \
reports/heart_diagnostic_analysis.pdf \
docs/index.html

# 1. Download and extract data
data/raw/pretransformed_heart_disease.csv: scripts/1_download_decode_data.py
	python scripts/1_download_decode_data.py \
		--id=45 \
		--write-to=data/raw

# 2. Read, validate, and split data
data/processed/train_df.csv data/processed/test_df.csv: scripts/2_data_split_validate.py \
data/raw/pretransformed_heart_disease.csv
	python scripts/2_data_split_validate.py \
		--split=0.2 \
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

# Build HTML report and copy build to docs folder
reports/heart_diagnostic_analysis.html reports/heart_diagnostic_analysis.pdf: reports/heart_diagnostic_analysis.qmd \
reports/references.bib \
results/tables/model_metrics.csv \
data/processed/train_df.csv \
results/figures/categorical_distributions.png \
results/figures/numeric_distributions.png \
results/figures/correlation_matrix.png \
results/tables/cross_val_score.csv \
results/figures/confusion_matrix.png
	quarto render reports/heart_diagnostic_analysis.qmd --to html
	quarto render reports/heart_diagnostic_analysis.qmd --to pdf 

# Copy HTML report to docs folder as index.html
docs/index.html: reports/heart_diagnostic_analysis.html
	mkdir -p docs
	cp reports/heart_diagnostic_analysis.html docs/index.html

# Clean up analysis
clean:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf results/figures/categorical_distributions.png \
		results/figures/confusion_matrix.png \
		results/figures/correlation_matrix.png \
		results/figures/numeric_distributions.png
	rm -rf results/models/disease_pipeline.pickle
	rm -rf results/tables/correlation_matrix.csv \
		results/tables/cross_val_score.csv \
		results/tables/cross_val_std.csv \
		results/tables/high_correlations.csv \
		results/tables/model_metrics.csv
	rm -rf reports/heart_diagnostic_analysis.pdf \
		reports/heart_diagnostic_analysis.html \
        docs/index.html \
		reports/heart_diagnostic_analysis_files
