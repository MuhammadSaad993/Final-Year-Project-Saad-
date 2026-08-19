# Comparative Analysis and Interpretation of Machine Learning Models for Diabetes Prediction using CDC Health Indicators

A machine learning pipeline that predicts diabetes/prediabetes risk from the CDC Diabetes Health Indicators dataset, comparing Random Forest, XGBoost, and SVM classifiers, applying domain-driven feature engineering, and interpreting model decisions with SHAP and LIME. Built and run in Google Colab, with a Streamlit demo app for interactive risk prediction.

## Overview

The notebook walks through a full applied machine learning workflow:

1. **Data acquisition** — fetches the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (UCI ML Repository, ID 891) via `ucimlrepo`.
2. **Exploratory Data Analysis (EDA)** — inspects dataset shape and structure, checks for missing values, visualizes the target variable distribution (`Diabetes_binary`), box plots for continuous features (BMI, mental health days, physical health days) split by diabetes status, frequency distributions of key categorical features (blood pressure, cholesterol, general health, age, physical activity, difficulty walking, education, income), and a full correlation heatmap ranked against the target.
3. **Data preprocessing & class balancing** — stratified subsampling to a manageable size, an isolated held-out test split, and SMOTE+ENN resampling applied only to the train/validation pool to correct class imbalance without leaking into the test set.
4. **Model initialization & training** — trains Random Forest, XGBoost, and SVM classifiers with class-imbalance corrections (`class_weight='balanced'`, `scale_pos_weight`), timing each model's training.
5. **Model evaluation** — accuracy, precision, recall, F1, ROC-AUC, precision-recall AUC, confusion matrices, and ROC curves compared across all three models.
6. **Advanced feature engineering & retraining** — adds domain-driven interaction features (a metabolic-risk proxy from high blood pressure and cholesterol, a sedentary-obesity index, an age-health interaction term, and a socioeconomic vulnerability index), then retrains and re-evaluates the models on the enriched feature set.
7. **Model interpretability** — global and local explainability using SHAP (TreeExplainer, feature importance bar chart, beeswarm summary plot) and LIME for the Random Forest model.
8. **Deployment demo** — a Streamlit app (`app.py`, generated from the notebook) that retrains the pipeline end-to-end and serves an interactive diabetes risk prediction interface, runnable directly from Colab via a proxied port.

## Dataset

- Source: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (UCI Machine Learning Repository, dataset ID `891`), derived from the CDC's Behavioral Risk Factor Surveillance System (BRFSS).
- Target: `Diabetes_binary` — `0` (healthy) vs. `1` (diabetic/prediabetic).
- Features include health indicators such as high blood pressure, high cholesterol, BMI, general/mental/physical health, physical activity, difficulty walking, education, and income.
- The notebook performs stratified subsampling (default 20,000 rows) to keep SVM training computationally feasible, with a pure, unmanipulated held-out test set.

## Requirements

The notebook is designed to run in **Google Colab**. Key dependencies:

```
ucimlrepo
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
lime
streamlit
joblib
```

## Setup & Usage

1. Open the notebook in Google Colab and run the cells in order.
2. The CDC Diabetes Health Indicators dataset is fetched automatically via `ucimlrepo` — no manual download or credentials needed.
3. Run the EDA cells to reproduce the target distribution, box plots, frequency distributions, and correlation heatmap (saved as PNG files).
4. Run the preprocessing/class-balancing, model training, and evaluation cells to reproduce the Random Forest, XGBoost, and SVM comparison.
5. Run the feature engineering and retraining cells to see the effect of the engineered interaction features on model performance.
6. Run the interpretability cells to generate SHAP and LIME explanations for the Random Forest model.
7. **Streamlit demo**: the final cells install Streamlit, write out `app.py`, and launch it on port `8501`, exposing a public URL via Colab's port proxy. The app retrains the pipeline on first run and then serves interactive predictions.

## Results

The notebook prints and visualizes:

- Target class distribution and imbalance ratio
- Comparative metrics table (accuracy, precision, recall, F1, ROC-AUC) across Random Forest, XGBoost, and SVM, before and after feature engineering
- Confusion matrices and ROC curves for each model
- SHAP global feature importance and beeswarm plots, plus LIME local explanations

Exact numbers depend on the training run and are printed at execution time rather than hardcoded here.

## Project Structure

```
.
├── notebook.ipynb              # Main notebook: EDA, preprocessing, training, evaluation, interpretability
├── app.py                      # Streamlit inference app (generated by the notebook)
├── target_distribution.png     # Target class distribution plot
├── box_plots.png               # Box plots of continuous features by diabetes status
├── frequency_distributions.png # Frequency distributions of categorical features
├── correlation_heatmap.png     # Correlation heatmap of all features vs. target
└── rf_shap_global_bar.png      # SHAP global feature importance (Random Forest)
```

## Notes

- This project uses survey-derived health indicator data for a machine learning demonstration and is **not** a certified diagnostic tool. Predictions should not be used for real clinical decision-making.
- The notebook was authored for the Colab environment; some cells (package installs, Colab port proxying) will need adjustment to run outside Colab.
