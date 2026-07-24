from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.generics import ListAPIView

from .models import Transaction, FraudPrediction
from .serializers import (
    ScoreRequestSerializer,
    FraudPredictionSerializer,
)
from . import ml_model


@api_view(['POST'])
def score_transaction(request):
    """
    POST /api/score/
    Body: {"amount": 120.50, "features": {"V1": -1.36, "V2": -0.07, ...}}

    Scores a transaction, stores it + its prediction, returns the result.
    """
    serializer = ScoreRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    transaction = Transaction.objects.create(
        amount=serializer.validated_data['amount'],
        features=serializer.validated_data['features'],
    )

    try:
        fraud_score = ml_model.score_transaction(serializer.validated_data['features'])
    except FileNotFoundError as e:
        return Response({'error': str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    prediction = FraudPrediction.objects.create(
        transaction=transaction,
        fraud_score=fraud_score,
        is_flagged=fraud_score >= settings.FRAUD_SCORE_THRESHOLD,
    )

    return Response(
        FraudPredictionSerializer(prediction).data,
        status=status.HTTP_201_CREATED,
    )


class FlaggedTransactionsView(ListAPIView):
    """GET /api/flagged/ — list all transactions currently flagged as fraud."""
    serializer_class = FraudPredictionSerializer

    def get_queryset(self):
        return FraudPrediction.objects.filter(is_flagged=True).order_by('-predicted_at')


@api_view(['POST'])
def submit_feedback(request, prediction_id):
    """
    POST /api/predictions/<id>/feedback/
    Body: {"feedback": "confirmed_fraud"} or {"feedback": "false_positive"}

    Simulates the human-in-the-loop feedback step of a real fraud system.
    """
    try:
        prediction = FraudPrediction.objects.get(pk=prediction_id)
    except FraudPrediction.DoesNotExist:
        return Response({'error': 'Prediction not found'}, status=status.HTTP_404_NOT_FOUND)

    feedback = request.data.get('feedback')
    if feedback not in dict(FraudPrediction.FEEDBACK_CHOICES):
        return Response({'error': 'Invalid feedback value'}, status=status.HTTP_400_BAD_REQUEST)

    prediction.feedback = feedback
    prediction.save()
    return Response(FraudPredictionSerializer(prediction).data)
