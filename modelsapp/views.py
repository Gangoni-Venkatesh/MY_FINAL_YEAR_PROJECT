# modelsapp/views.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
)

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404
from django.views.decorators.cache import never_cache

from .models import MLModel
from datasets.models import BlockchainDataset

BASE_DIR = Path(__file__).resolve().parent.parent


@login_required
def model_list(request):
    """
    List all active models using MLModel table.
    This is what drives the 'Trained Models' page.
    """
    models_qs = MLModel.objects.filter(is_active=True).order_by('name', '-created_at')

    # For compatibility with your existing template that loops over 'detailed_metrics'
    detailed_metrics = []
    for m in models_qs:
        detailed_metrics.append({
            'model': m,
            'pr_auc': m.pr_auc,
            'accuracy': m.accuracy,
            # Optional placeholders if template expects them
            'precision': None,
            'recall': None,
            'f1': None,
            'confusion_matrix': None,
            'pr_curve': None,
        })

    return render(request, 'modelsapp/model_list.html', {
        'models': models_qs,
        'detailed_metrics': detailed_metrics,
    })


@login_required
@never_cache
def model_detail(request, pk):
    """
    Detailed metrics for a single model: confusion matrix, classification report,
    PR curve, and metric bar chart.
    Dynamically calculates metrics using the most recent dataset.
    """
    model = get_object_or_404(MLModel, pk=pk, is_active=True)
    model_path = BASE_DIR / 'ml' / 'models' / model.model_file

    if not model_path.exists():
        return render(request, 'modelsapp/model_not_found.html', {'model': model})

    data = joblib.load(model_path)
    scaler = data['scaler']
    model_clf = data['model']

    FEATURE_COLS = [
        'block_interval',
        'block_size',
        'fee_rate',
        'difficulty',
        'hash_rate',
        'mempool_tx_count',
    ]

    # Try to load the most recent uploaded dataset by the current user
    latest_dataset = BlockchainDataset.objects.filter(
        uploaded_by=request.user
    ).order_by('-uploaded_at').first()
    
    dataset_name = "No dataset"
    sample_df = None
    
    if latest_dataset:
        try:
            sample_df = pd.read_csv(latest_dataset.csv_file.path)
            dataset_name = latest_dataset.name
        except Exception:
            pass
    
    # Fallback to sample data if no user dataset exists
    if sample_df is None or sample_df.empty:
        sample_path = BASE_DIR / 'ml' / 'sample_blockchain_data.csv'
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            dataset_name = "Sample dataset"
        else:
            return render(request, 'modelsapp/model_detail.html', {
                'model': model,
                'dataset_name': dataset_name,
                'confusion_matrix': None,
                'classification_report': None,
                'pr_auc': None,
                'pr_precision': [],
                'pr_recall': [],
                'y_pred_proba': [],
                'class1_precision': None,
                'class1_recall': None,
                'class1_f1': None,
                'overall_accuracy': None,
            })
    # Validate columns
    if not set(FEATURE_COLS + ['is_delay']).issubset(sample_df.columns):
        return render(request, 'modelsapp/model_detail.html', {
            'model': model,
            'dataset_name': dataset_name,
            'confusion_matrix': None,
            'classification_report': None,
            'pr_auc': None,
            'pr_precision': [],
            'pr_recall': [],
            'y_pred_proba': [],
            'class1_precision': None,
            'class1_recall': None,
            'class1_f1': None,
            'overall_accuracy': None,
        })

    # Drop rows with missing values
    sample_df = sample_df[FEATURE_COLS + ['is_delay']].dropna()
    
    X = sample_df[FEATURE_COLS].values
    y_true = sample_df['is_delay'].values.astype(int)

    if X.shape[0] == 0:
        return render(request, 'modelsapp/model_detail.html', {
            'model': model,
            'dataset_name': dataset_name,
            'confusion_matrix': None,
            'classification_report': None,
            'pr_auc': None,
            'pr_precision': [],
            'pr_recall': [],
            'y_pred_proba': [],
            'class1_precision': None,
            'class1_recall': None,
            'class1_f1': None,
            'overall_accuracy': None,
        })

    X_scaled = scaler.transform(X)
    y_pred_proba = model_clf.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    report['0']['f1_score'] = report['0']['f1-score']
    report['1']['f1_score'] = report['1']['f1-score']

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc_score = auc(recall, precision)

    class1_precision = report['1']['precision']
    class1_recall = report['1']['recall']
    class1_f1 = report['1']['f1_score']
    overall_accuracy = report['accuracy']

    return render(request, 'modelsapp/model_detail.html', {
        'model': model,
        'dataset_name': dataset_name,
        'total_samples': len(y_true),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'pr_auc': pr_auc_score,
        'pr_precision': precision.tolist(),
        'pr_recall': recall.tolist(),
        'y_pred_proba': y_pred_proba.tolist(),
        'class1_precision': class1_precision,
        'class1_recall': class1_recall,
        'class1_f1': class1_f1,
        'overall_accuracy': overall_accuracy,
    })
