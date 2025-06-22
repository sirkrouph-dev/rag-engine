"""
Django URL configuration for enhanced RAG Engine API.
"""
from django.urls import path
from .django_enhanced import (
    ChatAPIView, 
    BuildAPIView, 
    StatusAPIView, 
    HealthAPIView, 
    MetricsAPIView
)

urlpatterns = [
    path('chat/', ChatAPIView.as_view(), name='chat'),
    path('build/', BuildAPIView.as_view(), name='build'),
    path('status/', StatusAPIView.as_view(), name='status'),
    path('health/', HealthAPIView.as_view(), name='health'),
    path('health/ready/', HealthAPIView.as_view(), name='health-ready'),
    path('metrics/', MetricsAPIView.as_view(), name='metrics'),
]
