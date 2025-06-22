
# Credit Card Fraud Detection Pipeline

This repository provides an example pipeline for detecting fraudulent credit card transactions. It uses the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is **not** included in this repository; download `creditcard.csv` from Kaggle and place it in the `data/` directory before running the pipeline.

## Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`

These dependencies can be installed with:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

## Usage

1. Download `creditcard.csv` from Kaggle and put it in the `data/` folder.
2. Run the pipeline:

```bash
python3 src/fraud_detection_pipeline.py --csv data/creditcard.csv --results results
```

The script will preprocess the data, handle class imbalance with SMOTE, train a `RandomForestClassifier`, and save evaluation metrics and plots in the `results/` directory.

## Output

`results/` will contain:

- `metrics.txt` – AUC score and classification report with precision, recall, and f1-score.
- `roc_curve.png` – ROC curve plot.
- `confusion_matrix.png` – Confusion matrix visualization.

## Optional: PySpark

For large-scale processing, you can adapt the code to PySpark. A basic structure is provided in `src/pyspark_pipeline.py`. This script now also saves evaluation metrics and plots in the specified results directory.
=======
# Credit-Card-Fraud-Detection-Pipeline-Fintech-

## Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

