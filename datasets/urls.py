from django.urls import path
from . import views

# datasets/urls.py
app_name = 'datasets'

urlpatterns = [
    path('', views.dataset_list, name='list'),
    path('upload/', views.dataset_upload, name='upload'),
    path('<int:pk>/', views.dataset_detail, name='detail'),
    path('<int:pk>/train/', views.train_dataset, name='train_dataset'),
    path('<int:pk>/predictions/', views.dataset_predictions, name='predictions'),
]

