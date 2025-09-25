import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score
from sklearn.metrics import precision_recall_curve
import xgboost as xgb
from .config import load
from joblib import dump, load as jload

cfg = load()


def create_precision_features(df):
    """Create features specifically designed to improve precision (defensive: works with missing cols)."""
    df = df.copy()

    # Price-volume interaction (high volume + low price = better fade odds)
    if 'current_price' in df.columns and 'total_volume' in df.columns:
        # Avoid div-by-zero
        df['price_volume_ratio'] = df['total_volume'] / (df['current_price'].replace(0, np.nan).fillna(1e-6))

    # Halt timing score (early morning halts tend to fade more)
    if 'halt_hour' in df.columns:
        df['halt_timing_score'] = np.where(df['halt_hour'] <= 10, 1.0,
                                           np.where(df['halt_hour'] <= 14, 0.5, 0.2))

    # News-volume interaction (strong news + high volume = lower fade odds)
    if 'news_strength' in df.columns and 'total_volume' in df.columns:
        df['news_volume_interaction'] = df['news_strength'] * np.log1p(df['total_volume'].clip(lower=0))

    # VWAP distance ratio (how far from VWAP)
    if 'distance_from_vwap' in df.columns and 'current_price' in df.columns:
        df['vwap_distance_ratio'] = df['distance_from_vwap'] / (df['current_price'].replace(0, np.nan).fillna(1e-6))

    # Gap momentum (size of gap relative to price)
    if 'gap_size' in df.columns and 'current_price' in df.columns:
        df['gap_momentum'] = df['gap_size'] / (df['current_price'].replace(0, np.nan).fillna(1e-6))

    # Volume acceleration (recent volume vs average)
    if 'total_volume' in df.columns and 'avg_volume' in df.columns:
        df['volume_acceleration'] = df['total_volume'] / (df['avg_volume'].replace(0, np.nan).fillna(1e-6))

    # Fill any newly created NA with 0
    df = df.fillna(0).infer_objects(copy=False)
    return df


def prepare_training_data(features_df, label_col='fade_success', feature_columns=None):
    """
    Build training matrix X and label vector y.
    Returns: X (DataFrame), y (Series or None), feature_columns (list)
    """
    df = features_df.copy()

    # Base features
    base_feature_cols = [
        'pre_halt_return', 'pre_halt_vol_ratio', 'parabolic_pct', 'gap_from_prev_close', 'news_sentiment',
        'current_price', 'total_volume', 'avg_volume', 'volume_std',
        'halt_hour', 'halt_minute', 'distance_from_vwap', 'vwap_extended',
        'gap_size', 'hod_rejection', 'weakness_confirmed', 'news_strength',
        'multiple_halts', 'is_gap_down', 'multiple_gap_downs', 'is_biotech'
    ]

    # Precision-boosting features
    precision_features = [
        'price_volume_ratio', 'halt_timing_score', 'news_volume_interaction',
        'vwap_distance_ratio', 'gap_momentum', 'volume_acceleration'
    ]
    base_feature_cols.extend(precision_features)

    # Categorical features to one-hot
    categorical_features = ['price_bucket', 'time_of_day', 'category', 'gap_direction']

    # Ensure columns exist with safe defaults
    for c in base_feature_cols:
        if c not in df.columns:
            if c in ['vwap_extended', 'hod_rejection', 'weakness_confirmed', 'multiple_halts', 'is_gap_down', 'multiple_gap_downs', 'is_biotech']:
                df[c] = False
            else:
                df[c] = 0.0

    # Create precision-boosting features (safe)
    df = create_precision_features(df)

    # Ensure boolean columns are bool dtype
    boolean_cols = ['vwap_extended', 'hod_rejection', 'weakness_confirmed', 'multiple_halts', 'is_gap_down', 'multiple_gap_downs', 'is_biotech']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # One-hot categorical (keep deterministic columns order)
    all_categorical_dummies = []
    for cat_col in categorical_features:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col].astype(str), prefix=cat_col)
            dummies = dummies.sort_index(axis=1)  # stable ordering
            df = pd.concat([df, dummies], axis=1)
            all_categorical_dummies.extend(dummies.columns.tolist())

    # Final feature column list
    if feature_columns is None:
        # training mode: collect base + created categorical dummies
        feature_columns = list(dict.fromkeys(base_feature_cols + all_categorical_dummies))
    else:
        # prediction mode: ensure all expected cols exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0

    # Build X (guaranteed numeric)
    X = df[feature_columns].fillna(0.0)
    # Convert object columns to numeric where possible
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            except Exception:
                X[col] = 0.0
    X = X.astype(float)

    # y (if present)
    y = df[label_col].fillna(0).astype(int) if label_col in df.columns else None

    return X, y, feature_columns


def train_xgboost_classifier(X_train, y_train, X_val=None, y_val=None, params=None, target_precision=None):
    """
    Train an XGBoost classifier with sensible defaults and class imbalance handling.
    If validation set provided and target_precision is given, compute a decision_threshold on val set.
    Returns trained model (with attributes: feature_columns set by caller, decision_threshold possibly set).
    """
    params = params or cfg.get('model_params', {}).get('xgboost', {})

    # Defensive check: need at least 2 classes
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        raise ValueError("y_train must contain at least two classes (0 and 1) to train a classifier.")

    # Handle class imbalance
    try:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg) / float(max(pos, 1))
    except Exception:
        scale_pos_weight = 1.0

    model = xgb.XGBClassifier(
        n_estimators=int(params.get('n_estimators', 3000)),
        max_depth=int(params.get('max_depth', 4)),
        learning_rate=float(params.get('learning_rate', 0.1)),
        subsample=float(params.get('subsample', 0.8)),
        colsample_bytree=float(params.get('colsample_bytree', 0.8)),
        reg_alpha=float(params.get('reg_alpha', 0.5)),
        reg_lambda=float(params.get('reg_lambda', 1.5)),
        min_child_weight=float(params.get('min_child_weight', 1)),
        max_delta_step=float(params.get('max_delta_step', 1)),
        gamma=float(params.get('gamma', 0.5)),
        scale_pos_weight=float(params.get('scale_pos_weight', scale_pos_weight)),
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42,
        verbosity=1
    )

    # Fit with optional early stopping when validation provided
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['early_stopping_rounds'] = int(params.get('early_stopping_rounds', 50))
        fit_kwargs['verbose'] = True

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        # fallback if xgboost version complains about kwargs
        model.fit(X_train, y_train)
    except Exception:
        model.fit(X_train, y_train)

    # Optionally pick a probability threshold to hit target precision on validation set
    if (X_val is not None and y_val is not None) and (target_precision is not None):
        try:
            proba_val = model.predict_proba(X_val)[:, 1]
            thresh = select_threshold_for_precision(y_val, proba_val, target_precision=target_precision)
            setattr(model, 'decision_threshold', float(thresh))
        except Exception:
            setattr(model, 'decision_threshold', 0.5)
    else:
        # default decision threshold (more aggressive than 0.5 to prioritize precision)
        setattr(model, 'decision_threshold', float(params.get('default_decision_threshold', 0.35)))

    # Save model and metadata
    try:
        out_dir = cfg.get('output_dir', 'outputs')
        model_path = f"{out_dir.rstrip('/')}/xgb_fade_model.joblib"
        # Attach feature_columns if provided (caller should set it)
        # dump model
        dump(model, model_path)
    except Exception:
        # swallow saving errors
        pass

    return model


def select_threshold_for_precision(y_true, proba, target_precision=0.7):
    """
    Choose the largest threshold that yields precision >= target_precision on given set.
    If none found, fall back to threshold maximizing F1.
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        # precision and recall arrays length = len(thresholds) + 1
        # thresholds correspond to precision[:-1], recall[:-1]
        if len(thresholds) == 0:
            return 0.5

        # Find candidate thresholds where precision >= target_precision
        valid_idx = np.where(precision[:-1] >= target_precision)[0]
        if len(valid_idx) > 0:
            # choose highest threshold among valid (more conservative)
            candidates = thresholds[valid_idx]
            return float(np.max(candidates))

        # otherwise pick threshold that maximizes F1 on thresholds
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1_scores))
        return float(thresholds[best_idx])
    except Exception:
        return 0.5


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and print standard diagnostics. Returns dict with accuracy and roc_auc (or None).
    Uses model.decision_threshold if present.
    """
    proba = model.predict_proba(X_test)[:, 1]
    # threshold = getattr(model, 'decision_threshold', 0.35)
    threshold = 0.75
    pred = (proba >= threshold).astype(int)

    print("Decision threshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, pred))
    try:
        roc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None
        if roc is not None:
            print("ROC AUC:", roc)
    except Exception:
        roc = None

    print(classification_report(y_test, pred, digits=4))
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc
    }
    return metrics
