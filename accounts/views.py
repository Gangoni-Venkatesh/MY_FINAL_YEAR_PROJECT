from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.views.decorators.cache import never_cache
from django.http import JsonResponse
from django.contrib.auth.models import User

from .forms import UserRegisterForm, LoginForm
from modelsapp.models import MLModel
from datasets.models import BlockchainDataset
from predictions.models import Prediction


def register_view(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, 'Registration successful. Please log in.')
            return redirect('accounts:login')
    else:
        form = UserRegisterForm()
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('accounts:dashboard')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        messages.info(request, 'You have been logged out.')
        return redirect('home')
    return redirect('home')


@login_required
@never_cache
def dashboard_view(request):
    # Redirect superusers to admin dashboard
    if request.user.is_superuser:
        return redirect('accounts:admin_dashboard')
    
    # Get dynamic stats for regular users
    active_models = MLModel.objects.filter(is_active=True).count()
    user_datasets = BlockchainDataset.objects.filter(uploaded_by=request.user).count()
    user_predictions = Prediction.objects.filter(user=request.user).count()
    
    # Calculate average accuracy from active models
    models = MLModel.objects.filter(is_active=True)
    avg_accuracy = 0
    if models.exists():
        total_accuracy = sum([m.accuracy for m in models if m.accuracy])
        avg_accuracy = round(total_accuracy / models.count(), 1) if models.count() > 0 else 0
    
    context = {
        'active_models': active_models,
        'user_datasets': user_datasets,
        'user_predictions': user_predictions,
        'avg_accuracy': avg_accuracy,
    }
    
    return render(request, 'accounts/dashboard.html', context)


@login_required
def get_dashboard_stats(request):
    """API endpoint to fetch updated dashboard stats for real-time updates"""
    active_models = MLModel.objects.filter(is_active=True).count()
    user_datasets = BlockchainDataset.objects.filter(uploaded_by=request.user).count()
    user_predictions = Prediction.objects.filter(user=request.user).count()
    
    # Calculate average accuracy
    models = MLModel.objects.filter(is_active=True)
    avg_accuracy = 0
    if models.exists():
        total_accuracy = sum([m.accuracy for m in models if m.accuracy])
        avg_accuracy = round(total_accuracy / models.count(), 1) if models.count() > 0 else 0
    
    return JsonResponse({
        'active_models': active_models,
        'user_datasets': user_datasets,
        'user_predictions': user_predictions,
        'avg_accuracy': avg_accuracy,
    })


@login_required
def admin_dashboard_view(request):
    """Admin-only dashboard with user management"""
    if not request.user.is_superuser:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('accounts:dashboard')
    
    # Get all users
    all_users = User.objects.all().order_by('-date_joined')
    total_users = all_users.count()
    staff_users = all_users.filter(is_staff=True).count()
    superusers = all_users.filter(is_superuser=True).count()
    active_users = all_users.filter(is_active=True).count()
    
    # Get system statistics
    total_datasets = BlockchainDataset.objects.count()
    total_models = MLModel.objects.count()
    total_predictions = Prediction.objects.count()
    
    context = {
        'users': all_users,
        'total_users': total_users,
        'staff_users': staff_users,
        'superusers': superusers,
        'active_users': active_users,
        'total_datasets': total_datasets,
        'total_models': total_models,
        'total_predictions': total_predictions,
    }
    
    return render(request, 'accounts/admin_dashboard.html', context)


@login_required
def manage_user(request, user_id):
    """Manage individual user permissions and status"""
    if not request.user.is_superuser:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('accounts:dashboard')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('accounts:admin_dashboard')
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'toggle_status':
            user.is_active = not user.is_active
            user.save()
            status = "activated" if user.is_active else "deactivated"
            messages.success(request, f'User {user.username} has been {status}.')
        
        elif action == 'toggle_staff':
            user.is_staff = not user.is_staff
            user.save()
            status = "granted" if user.is_staff else "revoked"
            messages.success(request, f'Staff privileges {status} for {user.username}.')
        
        elif action == 'toggle_superuser':
            if user.id == request.user.id:
                messages.error(request, 'Cannot modify your own superuser status.')
            else:
                user.is_superuser = not user.is_superuser
                if user.is_superuser:
                    user.is_staff = True
                user.save()
                status = "granted" if user.is_superuser else "revoked"
                messages.success(request, f'Superuser privileges {status} for {user.username}.')
        
        elif action == 'delete':
            if user.id == request.user.id:
                messages.error(request, 'Cannot delete your own account.')
            else:
                username = user.username
                user.delete()
                messages.success(request, f'User {username} has been deleted.')
                return redirect('accounts:admin_dashboard')
        
        return redirect('accounts:manage_user', user_id=user.id)
    
    context = {'user': user}
    return render(request, 'accounts/manage_user.html', context)
