from django.contrib import admin
from .models import Transaction, FraudPrediction


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['id', 'amount', 'submitted_at']
    ordering = ['-submitted_at']


@admin.register(FraudPrediction)
class FraudPredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'transaction', 'fraud_score', 'is_flagged', 'feedback', 'predicted_at']
    list_filter = ['is_flagged', 'feedback']
    ordering = ['-predicted_at']
