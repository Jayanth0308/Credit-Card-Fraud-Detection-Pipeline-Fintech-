import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the credit card transactions dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame):
    """Split the dataset and apply preprocessing including scaling and SMOTE."""
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    sampler = SMOTE(random_state=42)

    preprocessor = Pipeline([
        ('scaler', scaler),
    ])

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)

    return X_resampled, X_test_scaled, y_resampled, y_test, preprocessor


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train the RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, results_dir: str):
    """Evaluate the model and save metrics and plots."""
    os.makedirs(results_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"AUC: {auc}\n")
        f.write(pd.DataFrame(report).transpose().to_string())
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:0.2f})')
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


def run(csv_path: str, results_dir: str = 'results'):
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, results_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')
    parser.add_argument('--csv', required=True, help='Path to creditcard.csv')
    parser.add_argument('--results', default='results', help='Directory to save results')
    args = parser.parse_args()
    run(args.csv, args.results)
