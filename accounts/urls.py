from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('manage-user/<int:user_id>/', views.manage_user, name='manage_user'),
    path('api/dashboard-stats/', views.get_dashboard_stats, name='dashboard_stats'),
]
