# Real-Time Fraud Detection System

A fraud/anomaly detection system that scores transactions in real time using
ML models trained on the Kaggle Credit Card Fraud dataset, served through a
Django REST API with PostgreSQL storage and a human feedback loop.

**Status:** 🚧 In progress — Week 1 of 8

## Why this project

Fraud detection is a real, widely-deployed ML problem with a genuinely hard
core challenge: the data is extremely imbalanced (fraud is ~0.17% of
transactions), so naive accuracy metrics are meaningless. This project's
goal isn't just "train a model that works" — it's to demonstrate:

- Proper handling of class imbalance (SMOTE, class weighting, threshold tuning)
- A comparison between supervised (Random Forest / XGBoost) and unsupervised
  (Isolation Forest) approaches
- Evaluation framed around business cost (false negatives vs false positives),
  not just accuracy
- A production-shaped system: API, storage, and a feedback loop — not just a
  Jupyter notebook

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Client /   │─────▶│  Django REST API │─────▶│  ML Model   │
│  Dashboard   │      │  (fraud_api app) │      │ (joblib)    │
└─────────────┘      └────────┬─────────┘      └─────────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │   PostgreSQL      │
                      │ Transaction +     │
                      │ FraudPrediction   │
                      └──────────────────┘
```

## Tech stack

- **Backend:** Django + Django REST Framework
- **Database:** PostgreSQL
- **ML:** scikit-learn, XGBoost, imbalanced-learn, SHAP (explainability)
- **Data:** [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Project plan

| Weeks | Focus | Status |
|-------|-------|--------|
| 1-2 | Data exploration, baseline model, imbalance handling, Django skeleton | 🚧 |
| 3-4 | Isolation Forest vs XGBoost/Random Forest comparison | ⬜ |
| 5-6 | Serving API, dashboard, feedback loop | ⬜ |
| 7-8 | Evaluation rigor (cost-based metrics, SHAP), deploy, polish README | ⬜ |

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd fraud-detection-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# edit .env with your local PostgreSQL credentials

# 5. Create the database (in psql)
createdb fraud_detection_db

# 6. Download the dataset (requires free Kaggle account + API token)
python scripts/download_data.py

# 7. Train the model
python scripts/train_model.py

# 8. Run migrations and start the server
cd fraud_project
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/score/` | Score a transaction, returns fraud probability + flag |
| GET | `/api/flagged/` | List all transactions currently flagged as fraud |
| POST | `/api/predictions/<id>/feedback/` | Mark a prediction as confirmed fraud / false positive |

Example request:
```bash
curl -X POST http://localhost:8000/api/score/ \
  -H "Content-Type: application/json" \
  -d '{"amount": 120.50, "features": {"V1": -1.36, "V2": -0.07, "...": 0.0}}'
```

## Evaluation (to be filled in as the project progresses)

- Precision / Recall / F1 per model
- ROC-AUC comparison: Logistic Regression vs Random Forest vs Isolation Forest
- Cost-based analysis: estimated $ impact of false negatives vs false positives
- SHAP explainability plots for flagged transactions

## Honest limitations

- The Kaggle dataset's features are PCA-anonymized (V1-V28), so this project
  focuses on the modeling/imbalance/serving problem rather than real-world
  feature engineering — a production system would need domain-specific
  features (velocity checks, merchant risk scores, device fingerprinting, etc.)
- This is a portfolio/learning project, not a production-hardened system —
  no auth on the API yet, no rate limiting, no real-time streaming ingestion.

## License

MIT
