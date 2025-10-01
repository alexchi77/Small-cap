# retrain.py
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

# === local imports (your repo structure) ===
from src.features import featurize_enriched, merge_features_into_enriched, load_enriched_csv
# from src.training import create_trading_based_labels
from src.model import prepare_training_data, train_xgboost_classifier, evaluate_model
import pickle

def create_trading_based_labels(enriched_df, feats_df):
    """
    Create fade success labels based on actual trading strategy criteria.
    Only rows meeting client criteria are kept and labeled.
    """
    merge_cols = ['ticker', 'halt_time', 'bars']
    if 'resume_time' in enriched_df.columns:
        merge_cols.append('resume_time')
    merged_df = feats_df.merge(
        enriched_df[merge_cols],
        on=['ticker', 'halt_time'],
        how='left'
    )

    def calculate_fade_success(row):
        """
        Calculate if a trade would have been successful based on client's strategy
        """
        bars = row['bars']
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return 0

        halt_time = pd.to_datetime(row.get('halt_time'), errors='coerce')
        resume_time_raw = row['resume_time'] if 'resume_time' in row.index else None
        resume_time = pd.to_datetime(resume_time_raw, errors='coerce') if resume_time_raw is not None else pd.NaT

        if pd.isna(resume_time):
            if 'timestamp' in bars.columns:
                if pd.notna(halt_time):
                    candidates = bars.loc[bars['timestamp'] >= halt_time, 'timestamp']
                    resume_time = candidates.iloc[0] if len(candidates) > 0 else bars['timestamp'].min()
                else:
                    resume_time = bars['timestamp'].min()
            else:
                return 0

        pre_bars = bars[bars['timestamp'] < resume_time].copy()
        post_bars = bars[bars['timestamp'] >= resume_time].copy()

        if post_bars.empty or pre_bars.empty:
            return 0

        total_vol = pre_bars['volume'].sum() if 'volume' in pre_bars.columns else 0
        vwap = ((pre_bars['close'] * pre_bars['volume']).sum() / total_vol) if total_vol > 0 else pre_bars['close'].iloc[-1]
        pre_halt_close = pre_bars['close'].iloc[-1]
        hod = pre_bars['high'].max()

        entry_found = False
        entry_price = None
        for i in range(1, min(len(post_bars), 20)):
            window = post_bars.iloc[:i+1]
            if len(window) < 2:
                continue

            current_close = window['close'].iloc[-1]
            current_high = window['high'].iloc[-1]
            prev_high = window['high'].iloc[-2]

            vwap_fail = current_close < vwap
            hod_rejection = current_high < hod
            lower_high = current_high < prev_high

            if sum([vwap_fail, hod_rejection, lower_high]) >= 2:
                entry_found = True
                entry_price = window['open'].iloc[-1]
                break

        if not entry_found:
            return 0

        category = row.get('category', 'UNKNOWN')
        if category == 'IPO_SPAC':
            target_price = pre_halt_close
        elif category in ['GAP_DOWN', 'MULTIPLE_GAP_DOWN']:
            target_price = min(vwap, pre_halt_close)
        elif category == 'NO_NEWS_PARABOLIC':
            target_price = pre_halt_close
        else:
            target_price = pre_halt_close * 0.98

        stop_price = hod * 1.02

        outcome_bars = post_bars.iloc[:min(len(post_bars), 480)]

        for idx, bar in outcome_bars.iterrows():
            if bar['high'] >= stop_price:
                return 0
            if bar['low'] <= target_price:
                return 1

        return 1 if outcome_bars['close'].iloc[-1] <= target_price else 0

    def meets_client_criteria(row):
        bars = row['bars']
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return False

        if row.get('current_price', 0) < 3:
            return False

        # if 'volume' in bars.columns:
        #     premarket_volume = bars['volume'].iloc[:30].sum() if len(bars) >= 30 else bars['volume'].sum()
        #     if premarket_volume < 1_000_000:
        #         return False

        if row.get('is_biotech', False):
            return False

        if row.get('current_price', 0) < 5 or row.get('current_price', 0) > 30:
            return False

        return True

    valid_mask = merged_df.apply(meets_client_criteria, axis=1)
    valid_df = merged_df.loc[valid_mask].copy()
    print(f"Records meeting client criteria: {len(valid_df)}/{len(merged_df)}")

    if len(valid_df) == 0:
        print("No valid records found!")
        return pd.DataFrame(columns=list(feats_df.columns) + ['fade_success'])

    print("Calculating labels for valid records...")
    valid_df['fade_success'] = valid_df.apply(calculate_fade_success, axis=1)

    common_cols = [c for c in feats_df.columns if c in valid_df.columns]
    result_df = valid_df[common_cols].copy()
    result_df['fade_success'] = valid_df['fade_success']

    success_rate = result_df['fade_success'].mean()
    print(f"Overall fade success rate: {success_rate:.2%}")
    print(f"Total positive samples: {(result_df['fade_success']==1).sum()}")
    print(f"Total negative samples: {(result_df['fade_success']==0).sum()}")

    return result_df

def main():
    # --- 1. Load enriched data ---
    # print("Loading enriched halts data...")
    # enriched = pd.read_pickle("outputs/halts_enriched.pkl")
    # # Or use CSV if that's your main source:
    # # enriched = load_enriched_csv("outputs/halts_meta.csv")

    # # --- 2. Feature engineering ---
    # print("Generating features...")
    # feats = featurize_enriched(enriched)
    # dataset = merge_features_into_enriched(enriched, feats)

    # # --- 3. Labeling ---
    # print("Creating trading-based labels...")
    # labeled = create_trading_based_labels(enriched, feats)
    # labeled.to_pickle('labeled.pkl')
    labeled = pd.read_pickle('labeled.pkl')
    # --- 4. Prepare data for training ---
    print("Preparing data...")
    X, y, feature_cols = prepare_training_data(labeled, label_col="fade_success")
    
    # --- 5. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # --- 6. Train model ---
    print("Training XGBoost model...")
    model = train_xgboost_classifier(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        target_precision=0.7,  # tweak based on risk tolerance
    )

    # --- 7. Evaluate ---
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation metrics:", metrics)

    # --- 8. Save model + metadata ---
    model_path = "outputs/xgb_fade_model_latest.joblib"
    dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    features_path = "outputs/xgb_fade_features_latest.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"✅ Feature list saved to {features_path}")


if __name__ == "__main__":
    main()
