# Fraud Detection System

A real-time transaction fraud scoring API. Trains ML models on the Kaggle
credit card fraud dataset and serves predictions through a Django REST API,
with results stored in PostgreSQL and a basic feedback loop for reviewing
flagged transactions.

## The problem

Fraud is about 0.17% of transactions in this dataset (492 out of 284,807).
That imbalance is the actual challenge here — a model that just predicts
"not fraud" every time gets 99.8% accuracy and is completely useless. So the
real work is in choosing the right metrics and understanding the tradeoff
between catching fraud (recall) and not drowning legitimate users in false
flags (precision).

Two models trained so far:

| Model | Precision (fraud) | Recall (fraud) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.05 | 0.92 | — |
| Random Forest | 0.93 | 0.79 | 0.96 |

Logistic Regression flags almost everything as fraud — high recall, useless
precision. Random Forest is the more usable model in practice: it misses
more fraud but the flags it does raise are actually trustworthy. Still need
to check whether Isolation Forest (unsupervised) can close that recall gap
without wrecking precision.

## Stack

Django + DRF, PostgreSQL, scikit-learn / XGBoost / imbalanced-learn, SHAP
for explainability later.

## Architecture

```
Client → Django REST API → ML model (joblib)
                ↓
          PostgreSQL (transactions + predictions)
```

## Setup

```bash
git clone https://github.com/asibulislam/fraud-detection-system.git
cd fraud-detection-system

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in your local PostgreSQL creds

python scripts/download_data.py   # needs a Kaggle API token in ~/.kaggle/
python scripts/train_model.py

cd fraud_project
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## API

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/api/score/` | Score a transaction, returns fraud probability |
| GET | `/api/flagged/` | List transactions currently flagged |
| POST | `/api/predictions/<id>/feedback/` | Mark a flag as confirmed fraud / false positive |

```bash
curl -X POST http://localhost:8000/api/score/ \
  -H "Content-Type: application/json" \
  -d '{"amount": 120.50, "features": {"V1": -1.36, "V2": -0.07}}'
```

## What's next

- Isolation Forest for an unsupervised comparison against the two models above
- SMOTE / threshold tuning to see if Logistic Regression's precision can be
  salvaged
- SHAP plots so flagged transactions come with an explanation, not just a
  score
- A cost-based framing for the metrics — a false negative and a false
  positive don't cost the same thing in a real fraud system, and the eval
  should reflect that instead of just reporting precision/recall in a
  vacuum

## Limitations

The dataset's features (V1–V28) are PCA-anonymized, so there's no real
feature engineering here — a production system would need transaction
velocity, merchant risk, device fingerprinting, etc. This is a learning
project, not a deployable one: no auth on the API, no rate limiting, no
streaming ingestion.

## License

MIT
