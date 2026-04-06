# predictions/views.py
import joblib
import numpy as np
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib import messages
from django.views.decorators.cache import never_cache

from .forms import PredictionInputForm
from .models import Prediction

BASE_DIR = Path(__file__).resolve().parent.parent


def load_sklearn_model(path):
    data = joblib.load(path)
    return data['scaler'], data['model'], data['features']


@login_required
@never_cache
def prediction_create(request):
    result = None
    if request.method == 'POST':
        form = PredictionInputForm(request.POST)
        if form.is_valid():
            ml_model = form.cleaned_data['model']
            block_interval = form.cleaned_data['block_interval']
            block_size = form.cleaned_data['block_size']
            fee_rate = form.cleaned_data['fee_rate']
            difficulty = form.cleaned_data['difficulty']
            hash_rate = form.cleaned_data['hash_rate']
            mempool_tx_count = form.cleaned_data['mempool_tx_count']

            X = np.array([[block_interval, block_size, fee_rate,
                           difficulty, hash_rate, mempool_tx_count]])

            model_path = BASE_DIR / 'ml' / 'models' / ml_model.model_file

            try:
                scaler, model_clf, _ = load_sklearn_model(model_path)
            except Exception as e:
                messages.error(request, f"Could not load model file: {e}")
                return render(request, 'predictions/predict.html', {
                    'form': form,
                    'result': None,
                })

            X_scaled = scaler.transform(X)
            proba = float(model_clf.predict_proba(X_scaled)[0, 1])
            is_delay = proba >= 0.5

            if is_delay:
                recommendation = "High delay risk. Consider waiting or increasing fee rate."
            else:
                recommendation = "Low delay risk. It is a good time to broadcast transaction."

            result = Prediction.objects.create(
                user=request.user,
                model=ml_model,
                block_interval=block_interval,
                block_size=block_size,
                fee_rate=fee_rate,
                difficulty=difficulty,
                hash_rate=hash_rate,
                mempool_tx_count=mempool_tx_count,
                is_delay=is_delay,
                probability=proba,
                recommendation=recommendation,
            )
            messages.success(request, 'Prediction generated.')
    else:
        form = PredictionInputForm()

    return render(request, 'predictions/predict.html', {
        'form': form,
        'result': result,
    })


@login_required
@never_cache
def prediction_history(request):
    preds = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'predictions/history.html', {'predictions': preds})
