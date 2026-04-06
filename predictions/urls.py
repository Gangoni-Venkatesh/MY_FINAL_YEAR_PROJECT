from django.urls import path
from . import views

urlpatterns = [
    path('', views.prediction_create, name='create'),
    path('history/', views.prediction_history, name='history'),
]
