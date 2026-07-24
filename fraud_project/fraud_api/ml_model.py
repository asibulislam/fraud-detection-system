"""
Thin wrapper around the trained fraud detection model.

Loads a joblib-serialized sklearn/xgboost model from MODELS_DIR
(set in settings.ML_MODEL_DIR) and exposes a single `score()` function
the API view calls. Keeping this separate from views.py means you can
swap models (Isolation Forest -> XGBoost -> ensemble) without touching
the API layer.

Expected model file: models/fraud_model.joblib
Produced by: scripts/train_model.py (see project root /scripts)
"""
import joblib
from pathlib import Path
from django.conf import settings

_model = None
_model_path = Path(settings.ML_MODEL_DIR) / 'fraud_model.joblib'


def get_model():
    global _model
    if _model is None:
        if not _model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {_model_path}. "
                f"Run scripts/train_model.py first."
            )
        _model = joblib.load(_model_path)
    return _model


def score_transaction(features: dict) -> float:
    """
    Takes a dict of feature_name -> value, returns a fraud probability (0-1).
    NOTE: feature order must match what the model was trained on —
    scripts/train_model.py should save the feature column order alongside
    the model so this stays consistent. Placeholder logic below;
    replace once train_model.py is implemented.
    """
    model = get_model()
    # Placeholder — real implementation will build a properly-ordered
    # feature vector from `features` before calling model.predict_proba
    import numpy as np
    feature_vector = np.array([list(features.values())])
    proba = model.predict_proba(feature_vector)[0][1]  # probability of class "fraud"
    return float(proba)
