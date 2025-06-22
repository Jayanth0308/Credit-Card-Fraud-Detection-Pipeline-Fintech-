"""PySpark version of the fraud detection pipeline (simplified)."""

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


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
    spark.stop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PySpark Fraud Detection Pipeline')
    parser.add_argument('--csv', required=True, help='Path to creditcard.csv')
    parser.add_argument('--results', default='results', help='Directory to save results')
    args = parser.parse_args()
    run(args.csv, args.results)
