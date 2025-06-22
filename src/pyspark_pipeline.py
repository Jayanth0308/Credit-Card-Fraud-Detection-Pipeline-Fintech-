"""PySpark version of the fraud detection pipeline (simplified)."""

import os

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


def run(csv_path: str, results_dir: str = 'results'):
    spark = SparkSession.builder.appName('fraud-detection').getOrCreate()
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    assembler = VectorAssembler(inputCols=[c for c in df.columns if c != 'Class'], outputCol='features')
    scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
    rf = RandomForestClassifier(labelCol='Class', featuresCol='scaledFeatures')

    pipeline = Pipeline(stages=[assembler, scaler, rf])
    model = pipeline.fit(df)

    predictions = model.transform(df)
    evaluator = BinaryClassificationEvaluator(labelCol='Class')
    auc = evaluator.evaluate(predictions)
    print('AUC:', auc)

    os.makedirs(results_dir, exist_ok=True)
    pdf = predictions.select('Class', 'prediction', 'probability').toPandas()
    y_true = pdf['Class'].astype(int)
    y_pred = pdf['prediction'].astype(int)
    y_prob = pdf['probability'].apply(lambda x: float(x[1]))

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"AUC: {auc_score}\n")
        f.write(pd.DataFrame(report).transpose().to_string())
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    spark.stop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PySpark Fraud Detection Pipeline')
    parser.add_argument('--csv', required=True, help='Path to creditcard.csv')
    parser.add_argument('--results', default='results', help='Directory to save results')
    args = parser.parse_args()
    run(args.csv, args.results)
