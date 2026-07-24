from rest_framework import serializers
from .models import Transaction, FraudPrediction


class TransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = ['id', 'amount', 'features', 'submitted_at']


class FraudPredictionSerializer(serializers.ModelSerializer):
    transaction = TransactionSerializer(read_only=True)

    class Meta:
        model = FraudPrediction
        fields = [
            'id', 'transaction', 'fraud_score', 'is_flagged',
            'model_version', 'predicted_at', 'feedback',
        ]


class ScoreRequestSerializer(serializers.Serializer):
    """Input payload for POST /api/score/"""
    amount = serializers.DecimalField(max_digits=12, decimal_places=2)
    features = serializers.DictField(
        child=serializers.FloatField(),
        help_text="e.g. {'V1': -1.36, 'V2': -0.07, ...} matching your trained model's feature names"
    )
