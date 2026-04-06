import csv
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.cache import never_cache

from predictions.models import Prediction
from modelsapp.models import MLModel
from datasets.models import BlockchainDataset

BASE_DIR = Path(__file__).resolve().parent.parent

@login_required
@never_cache
def reports_overview(request):
    """
    Comprehensive reports dashboard showing all model predictions and performance.
    """
    # Get selected model from query parameter
    selected_model = request.GET.get('model', 'MLE')
    
    # Get all active models
    models = MLModel.objects.filter(is_active=True).order_by('name')
    
    # Get model metrics from CSV
    metrics_path = BASE_DIR / 'ml' / 'models' / 'model_metrics.csv'
    metrics_rows = []
    if metrics_path.exists():
        with open(metrics_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                metrics_rows.append({
                    'model': r['model'],
                    'pr_auc': float(r['pr_auc']),
                    'accuracy': float(r['accuracy']),
                })
    
    # Get latest dataset for evaluation
    latest_dataset = BlockchainDataset.objects.filter(
        uploaded_by=request.user
    ).order_by('-uploaded_at').first()
    
    dataset_name = "No dataset"
    all_model_results = []
    selected_model_detail = None
    
    FEATURE_COLS = [
        'block_interval',
        'block_size',
        'fee_rate',
        'difficulty',
        'hash_rate',
        'mempool_tx_count',
    ]
    TARGET_COL = 'is_delay'
    
    if latest_dataset:
        try:
            df = pd.read_csv(latest_dataset.csv_file.path)
            dataset_name = latest_dataset.name
            
            # Drop NaNs
            df = df[FEATURE_COLS + [TARGET_COL]].dropna()
            
            if len(df) > 0:
                X = df[FEATURE_COLS].values
                y_actual = df[TARGET_COL].values.astype(int)
                
                # Get predictions from each model
                for model in models:
                    try:
                        model_path = BASE_DIR / 'ml' / 'models' / model.model_file
                        if model_path.exists():
                            model_data = joblib.load(model_path)
                            scaler = model_data['scaler']
                            clf = model_data['model']
                            
                            X_scaled = scaler.transform(X)
                            proba = clf.predict_proba(X_scaled)[:, 1]
                            y_pred = (proba >= 0.5).astype(int)
                            
                            accuracy = float(np.mean(y_actual == y_pred)) * 100
                            delayed = int(np.sum(y_pred == 1))
                            ontime = int(np.sum(y_pred == 0))
                            correct = int(np.sum(y_actual == y_pred))
                            
                            result_data = {
                                'model_name': model.get_name_display(),
                                'model_code': model.name,
                                'total_predictions': len(y_pred),
                                'delayed_count': delayed,
                                'ontime_count': ontime,
                                'correct_count': correct,
                                'accuracy': accuracy,
                                'pr_auc': model.pr_auc,
                                'probabilities': proba.tolist(),
                                'predictions': y_pred.tolist(),
                                'actual': y_actual.tolist(),
                            }
                            
                            all_model_results.append(result_data)
                            
                            # Store details for selected model
                            if model.name == selected_model:
                                selected_model_detail = result_data
                    except Exception as e:
                        print(f"Error loading model {model.name}: {e}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    # Sort model results by accuracy (descending) BEFORE selecting, so we can reselect after sort
    all_model_results = sorted(all_model_results, key=lambda x: x['accuracy'], reverse=True)
    
    # Find the selected model in the sorted results list
    selected_model_detail = None
    for result in all_model_results:
        if result['model_code'] == selected_model:
            selected_model_detail = result
            break
    
    # If selected model not found, use first available
    if not selected_model_detail and all_model_results:
        selected_model_detail = all_model_results[0]
        selected_model = all_model_results[0]['model_code']
    
    # Get user's prediction history
    user_predictions = Prediction.objects.filter(
        user=request.user
    ).select_related('model').order_by('-created_at')[:20]
    
    # Summary statistics
    total_predictions = Prediction.objects.filter(user=request.user).count()
    delayed_predictions = Prediction.objects.filter(
        user=request.user, is_delay=True
    ).count()
    
    best_model_name = all_model_results[0]['model_code'] if all_model_results else '-'
    
    context = {
        'metrics': metrics_rows,
        'models': models,
        'all_model_results': all_model_results,
        'best_model_name': best_model_name,
        'selected_model': selected_model,
        'selected_model_detail': selected_model_detail,
        'dataset_name': dataset_name,
        'user_predictions': user_predictions,
        'total_predictions': total_predictions,
        'delayed_predictions': delayed_predictions,
    }
    
    return render(request, 'reports/overview.html', context)
