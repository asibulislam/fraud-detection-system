"""
Train the fraud detection model(s) and save the best one to models/fraud_model.joblib

This is a skeleton mapped to Weeks 1-4 of the project plan:
  Week 1-2: baseline model + imbalance handling
  Week 3-4: Isolation Forest + XGBoost comparison

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
# from imblearn.over_sampling import SMOTE
# import xgboost as xgb

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'creditcard.csv'
MODEL_OUT = Path(__file__).resolve().parent.parent / 'models' / 'fraud_model.joblib'


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Class'])
    y = df['Class']  # 1 = fraud, 0 = legit
    return X, y


def train_baseline(X_train, y_train, X_test, y_test):
    """Week 1-2: simple baseline to get a number to beat."""
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("=== Baseline Logistic Regression ===")
    print(classification_report(y_test, preds))
    return model


def train_random_forest(X_train, y_train, X_test, y_test):
    """Week 3-4: supervised comparison model."""
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


def train_isolation_forest(X_train):
    """Week 3-4: unsupervised comparison — doesn't use labels."""
    model = IsolationForest(contamination=0.0017, random_state=42, n_jobs=-1)
    model.fit(X_train)
    return model


# TODO (Week 3-4): add XGBoost training + compare all 3 models on same test set
# TODO (Week 5-6): add SMOTE variant, threshold tuning, save eval report to /notebooks
# TODO (Week 7-8): add SHAP explainability output


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_baseline(X_train, y_train, X_test, y_test)
    best_model = train_random_forest(X_train, y_train, X_test, y_test)
    # isolation_model = train_isolation_forest(X_train)  # compare separately — unsupervised

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == '__main__':
    main()
