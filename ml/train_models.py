# ml/train_models.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'ml' / 'sample_blockchain_data.csv'
MODEL_DIR = BASE_DIR / 'ml' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'block_interval',
    'block_size',
    'fee_rate',
    'difficulty',
    'hash_rate',
    'mempool_tx_count',
]

TARGET_COL = 'is_delay'


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


def train_variant(X_train, y_train, X_test, y_test, scaler, variant):
    """
    Train three slightly different logistic regressions so we can still
    compare 'MLE', 'Bayes_HMC', 'Bayes_Gibbs' without Bayesian libs.
    """
    if variant == 'MLE':
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif variant == 'Bayes_HMC':
        # Stronger regularization, different C
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
    elif variant == 'Bayes_Gibbs':
        # Even stronger regularization, different solver
        clf = LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.2, solver='liblinear'
        )
    else:
        raise ValueError("Unknown variant")

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, proba)
    acc = accuracy_score(y_test, y_pred)

    filename_map = {
        'MLE': 'logreg_mle.pkl',
        'Bayes_HMC': 'logreg_bayes_hmc.pkl',
        'Bayes_Gibbs': 'logreg_bayes_gibbs.pkl',
    }
    joblib.dump(
        {'scaler': scaler, 'model': clf, 'features': FEATURE_COLS},
        MODEL_DIR / filename_map[variant]
    )

    return pr_auc, acc, filename_map[variant]


def main():
    print("Loading data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows = []
    for variant in ['MLE', 'Bayes_HMC', 'Bayes_Gibbs']:
        print(f"Training {variant} logistic regression...")
        pr_auc, acc, fname = train_variant(
            X_train_scaled, y_train, X_test_scaled, y_test, scaler, variant
        )
        print(f"{variant} - PR AUC: {pr_auc:.4f}, Accuracy: {acc:.4f}")
        rows.append({'model': variant, 'pr_auc': pr_auc, 'accuracy': acc})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(MODEL_DIR / 'model_metrics.csv', index=False)
    print("Saved models and metrics in ml/models/.")


if __name__ == '__main__':
    main()
