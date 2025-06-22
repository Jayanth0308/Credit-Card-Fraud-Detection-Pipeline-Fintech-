# Credit Card Fraud Detection Pipeline

A sample machine learning pipeline for detecting fraudulent credit card transactions. It uses the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is **not** included; download `creditcard.csv` from Kaggle and place it in `data/`.

## Features

- Scaling and SMOTE to handle class imbalance
- RandomForest classifier with evaluation metrics
- Metrics and plots saved to a chosen results directory
- Optional PySpark script for processing larger datasets

## Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

## Running the pipeline

```bash
python3 src/fraud_detection_pipeline.py --csv data/creditcard.csv --results results
```

The `results/` directory will contain:

- `metrics.txt` – AUC and classification report
- `roc_curve.png` – ROC curve
- `confusion_matrix.png` – Confusion matrix plot

#### PySpark support (optional)

A simplified PySpark version lives in `src/pyspark_pipeline.py` and saves metrics and plots with a `pyspark_` prefix:

```bash
python3 src/pyspark_pipeline.py --csv data/creditcard.csv --results results
```
