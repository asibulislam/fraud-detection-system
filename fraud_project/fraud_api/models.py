from django.db import models


class Transaction(models.Model):
    """
    A single transaction submitted for fraud scoring.
    Feature fields are intentionally generic (feature_1..feature_n as JSON)
    so this works with PCA-anonymized datasets (like Kaggle's creditcard.csv)
    as well as raw feature sets later.
    """
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    features = models.JSONField(help_text="Raw model input features as key-value pairs")
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transaction #{self.pk} - amount={self.amount}"


class FraudPrediction(models.Model):
    """
    Model output for a given transaction: fraud score + flag,
    plus a slot for human feedback (confirmed fraud / false positive)
    to simulate a real feedback loop.
    """
    FEEDBACK_CHOICES = [
        ('unreviewed', 'Unreviewed'),
        ('confirmed_fraud', 'Confirmed Fraud'),
        ('false_positive', 'False Positive'),
    ]

    transaction = models.OneToOneField(
        Transaction, on_delete=models.CASCADE, related_name='prediction'
    )
    fraud_score = models.FloatField(help_text="Model's raw probability/anomaly score, 0-1")
    is_flagged = models.BooleanField(default=False)
    model_version = models.CharField(max_length=50, default='v1')
    predicted_at = models.DateTimeField(auto_now_add=True)
    feedback = models.CharField(
        max_length=20, choices=FEEDBACK_CHOICES, default='unreviewed'
    )

    def __str__(self):
        return f"Prediction for Txn #{self.transaction_id} - score={self.fraud_score:.3f}"
