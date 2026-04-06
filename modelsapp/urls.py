# modelsapp/urls.py
from django.urls import path
from . import views

app_name = 'modelsapp'

urlpatterns = [
    path('', views.model_list, name='list'),
    path('<int:pk>/', views.model_detail, name='detail'),  # new
]
