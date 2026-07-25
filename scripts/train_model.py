"""
Train the fraud detection model(s) and save the best one to models/fraud_model.joblib

Day 2: adds Isolation Forest as a third, unsupervised comparison point
against Logistic Regression and Random Forest.

Run:
  python scripts/train_model.py
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'creditcard.csv'
MODEL_OUT = Path(__file__).resolve().parent.parent / 'models' / 'fraud_model.joblib'


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Class'])
    y = df['Class']  # 1 = fraud, 0 = legit
    return X, y


def train_baseline(X_train, y_train, X_test, y_test):
    """Simple baseline to get a number to beat."""
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("=== Baseline Logistic Regression ===")
    print(classification_report(y_test, preds))
    return model


def train_random_forest(X_train, y_train, X_test, y_test):
    """Supervised comparison model."""
    model = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("=== Random Forest ===")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    return model


def train_isolation_forest(X_train, X_test, y_test):
    """
    Unsupervised comparison — doesn't use fraud labels during training,
    only learns what 'normal' transactions look like and flags outliers.
    """
    model = IsolationForest(contamination=0.0017, random_state=42, n_jobs=-1)
    model.fit(X_train)

    raw_preds = model.predict(X_test)
    preds = (raw_preds == -1).astype(int)

    print("=== Isolation Forest (unsupervised) ===")
    print(classification_report(y_test, preds))
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_baseline(X_train, y_train, X_test, y_test)
    best_model = train_random_forest(X_train, y_train, X_test, y_test)
    train_isolation_forest(X_train, X_test, y_test)

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == '__main__':
    main()
