from django.urls import path
from . import views

urlpatterns = [
    path('score/', views.score_transaction, name='score-transaction'),
    path('flagged/', views.FlaggedTransactionsView.as_view(), name='flagged-transactions'),
    path('predictions/<int:prediction_id>/feedback/', views.submit_feedback, name='submit-feedback'),
]
