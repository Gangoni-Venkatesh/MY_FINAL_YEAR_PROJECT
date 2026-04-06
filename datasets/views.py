# datasets/views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.views.decorators.cache import never_cache

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score

from .models import BlockchainDataset
from .forms import BlockchainDatasetForm
from modelsapp.models import MLModel


@login_required
def dataset_list(request):
    datasets = BlockchainDataset.objects.filter(
        uploaded_by=request.user
    ).order_by('-uploaded_at')
    return render(request, 'datasets/dataset_list.html', {'datasets': datasets})


@login_required
def dataset_upload(request):
    if request.method == 'POST':
        form = BlockchainDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            ds = form.save(commit=False)
            ds.uploaded_by = request.user
            ds.save()
            messages.success(request, 'Dataset uploaded successfully.')
            return redirect('datasets:list')
    else:
        form = BlockchainDatasetForm()
    return render(request, 'datasets/dataset_upload.html', {'form': form})


@login_required
def dataset_detail(request, pk):
    ds = get_object_or_404(BlockchainDataset, pk=pk, uploaded_by=request.user)
    return render(request, 'datasets/dataset_detail.html', {'dataset': ds})


@login_required
def train_dataset(request, pk):
    """
    Train three logistic regression variants on the selected dataset and
    update the MLModel table so that the Models page and Predict dropdown
    always show the latest models.
    """
    dataset = get_object_or_404(BlockchainDataset, pk=pk, uploaded_by=request.user)

    try:
        df = pd.read_csv(dataset.csv_file.path)
    except Exception as e:
        messages.error(request, f"Could not read CSV: {e}")
        return redirect('datasets:detail', pk=dataset.pk)

    FEATURE_COLS = [
        'block_interval',
        'block_size',
        'fee_rate',
        'difficulty',
        'hash_rate',
        'mempool_tx_count',
    ]
    TARGET_COL = 'is_delay'

    # 1) Check required columns
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        messages.error(
            request,
            f"CSV is missing required columns: {', '.join(missing)}. "
            "Expected columns: " + ', '.join(FEATURE_COLS + [TARGET_COL])
        )
        return redirect('datasets:detail', pk=dataset.pk)

    # 2) Drop NaNs
    df = df[FEATURE_COLS + [TARGET_COL]]
    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)

    if after_rows == 0:
        messages.error(
            request,
            "All rows contain NaN values. Please clean the CSV and upload again."
        )
        return redirect('datasets:detail', pk=dataset.pk)

    if after_rows < before_rows:
        messages.info(
            request,
            f"Dropped {before_rows - after_rows} rows with missing values before training."
        )

    # 3) Target 0/1
    try:
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    except Exception:
        messages.error(
            request,
            "Column 'is_delay' must contain only numeric 0 or 1."
        )
        return redirect('datasets:detail', pk=dataset.pk)

    # 4) Both classes
    if df[TARGET_COL].nunique() < 2:
        messages.error(
            request,
            "Column 'is_delay' must contain both classes 0 and 1 for training."
        )
        return redirect('datasets:detail', pk=dataset.pk)

    # 5) Enough rows
    if len(df) < 20:
        messages.error(
            request,
            "Dataset too small for reliable training (need at least 20 rows)."
        )
        return redirect('datasets:detail', pk=dataset.pk)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 6) Scale full data
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / 'ml' / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        ('MLE', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)),
        ('Bayes_HMC', LogisticRegression(max_iter=1000, class_weight='balanced', C=0.6)),
        ('Bayes_Gibbs', LogisticRegression(max_iter=1000, class_weight='balanced', C=0.3, solver='liblinear')),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for variant_name, base_clf in variants:
        pr_auc_scores = []
        acc_scores = []

        for train_idx, test_idx in skf.split(X_scaled_full, y):
            X_tr, X_te = X_scaled_full[train_idx], X_scaled_full[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf = LogisticRegression(
                max_iter=base_clf.max_iter,
                class_weight=base_clf.class_weight,
                C=base_clf.C,
                solver=base_clf.solver,
            )
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            y_pred = (proba >= 0.5).astype(int)

            pr_auc_scores.append(average_precision_score(y_te, proba))
            acc_scores.append(accuracy_score(y_te, y_pred))

        mean_pr_auc = float(np.mean(pr_auc_scores))
        mean_acc = float(np.mean(acc_scores))

        # Refit final model on all data
        final_clf = LogisticRegression(
            max_iter=base_clf.max_iter,
            class_weight=base_clf.class_weight,
            C=base_clf.C,
            solver=base_clf.solver,
        )
        final_clf.fit(X_scaled_full, y)

        filename = f'logreg_{variant_name.lower()}.pkl'
        joblib.dump(
            {'scaler': scaler_full, 'model': final_clf, 'features': FEATURE_COLS},
            MODEL_DIR / filename
        )

        rows.append({'model': variant_name, 'pr_auc': mean_pr_auc, 'accuracy': mean_acc})

        # 7) UPDATE / CREATE MLModel ROW
        MLModel.objects.update_or_create(
            name=variant_name,
            defaults={
                'version': 'v1',
                'pr_auc': mean_pr_auc,
                'accuracy': mean_acc,
                'model_file': filename,
                'is_active': True,
            }
        )

    # 8) Save metrics CSV
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(MODEL_DIR / 'model_metrics.csv', index=False)

    messages.success(
        request,
        f"Models retrained on {dataset.name}! "
        f"MLE PR‑AUC: {rows[0]['pr_auc']:.3f}, Accuracy: {rows[0]['accuracy']:.3f}"
    )

    return redirect('datasets:detail', pk=dataset.pk)


@login_required
@never_cache
def dataset_predictions(request, pk):
    """
    Show predictions on the entire dataset using the selected model.
    Displays which blocks are classified as delayed vs on-time.
    """
    dataset = get_object_or_404(BlockchainDataset, pk=pk, uploaded_by=request.user)

    # Get selected model from query parameter
    selected_model = request.GET.get('model', 'MLE')
    
    # Debug logging
    import sys
    print(f"[DEBUG] Selected model: {selected_model}", file=sys.stderr)
    
    # Available models
    model_options = {
        'MLE': {'name': 'MLE Logistic Regression', 'file': 'logreg_mle.pkl'},
        'Bayes_HMC': {'name': 'Bayesian HMC Logistic Regression', 'file': 'logreg_bayes_hmc.pkl'},
        'Bayes_Gibbs': {'name': 'Bayesian Gibbs Logistic Regression', 'file': 'logreg_bayes_gibbs.pkl'},
    }
    
    # Find the selected model file
    if selected_model in model_options:
        model_file = model_options[selected_model]['file']
        model_display_name = model_options[selected_model]['name']
    else:
        selected_model = 'MLE'
        model_file = model_options['MLE']['file']
        model_display_name = model_options['MLE']['name']
    
    print(f"[DEBUG] Using model file: {model_file}", file=sys.stderr)

    try:
        df = pd.read_csv(dataset.csv_file.path)
    except Exception as e:
        messages.error(request, f"Could not read CSV: {e}")
        return redirect('datasets:detail', pk=dataset.pk)

    FEATURE_COLS = [
        'block_interval',
        'block_size',
        'fee_rate',
        'difficulty',
        'hash_rate',
        'mempool_tx_count',
    ]
    TARGET_COL = 'is_delay'

    # Check required columns
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        messages.error(
            request,
            f"CSV is missing required columns: {', '.join(missing)}"
        )
        return redirect('datasets:detail', pk=dataset.pk)

    # Drop NaNs
    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    
    if len(df) == 0:
        messages.error(request, "No valid rows in dataset.")
        return redirect('datasets:detail', pk=dataset.pk)

    # Load selected model
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / 'ml' / 'models'
    model_path = MODEL_DIR / model_file

    try:
        model_data = joblib.load(model_path)
        scaler = model_data['scaler']
        model_clf = model_data['model']
    except Exception as e:
        messages.error(
            request,
            f"Could not load model. Please train models first. Error: {e}"
        )
        return redirect('datasets:detail', pk=dataset.pk)

    # Make predictions
    X = df[FEATURE_COLS].values
    y_actual = df[TARGET_COL].values.astype(int)
    
    X_scaled = scaler.transform(X)
    probas = model_clf.predict_proba(X_scaled)[:, 1]
    y_pred = (probas >= 0.5).astype(int)
    
    # Debug logging
    print(f"[DEBUG] Total samples: {len(y_pred)}", file=sys.stderr)
    print(f"[DEBUG] Predicted delayed: {np.sum(y_pred == 1)}", file=sys.stderr)
    print(f"[DEBUG] Predicted on-time: {np.sum(y_pred == 0)}", file=sys.stderr)
    print(f"[DEBUG] Sample probabilities: {probas[:5]}", file=sys.stderr)

    # Create results dataframe
    results_df = df.copy()
    results_df['actual_delay'] = y_actual
    results_df['predicted_delay'] = y_pred
    results_df['probability'] = probas
    results_df['correct'] = (y_actual == y_pred)

    # Calculate accuracy
    accuracy = float(np.mean(y_actual == y_pred))
    delayed_count = int(np.sum(y_pred == 1))
    ontime_count = int(np.sum(y_pred == 0))
    correct_count = int(np.sum(y_actual == y_pred))

    # Paginate results
    paginator = Paginator(results_df.to_dict('records'), 50)  # 50 rows per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'dataset': dataset,
        'page_obj': page_obj,
        'total_rows': len(results_df),
        'accuracy': accuracy,
        'delayed_count': delayed_count,
        'ontime_count': ontime_count,
        'correct_count': correct_count,
        'selected_model': selected_model,
        'model_display_name': model_display_name,
        'model_options': model_options,
    }

    return render(request, 'datasets/dataset_predictions.html', context)
