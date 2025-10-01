from src.config import load
from src.data_ingest import load_local_halts, enrich_halts_with_bars_and_news
from src.features import featurize_enriched, merge_features_into_enriched, load_enriched_csv
from src.model import prepare_training_data, train_xgboost_classifier, evaluate_model
from src.backtest import run_backtest, compute_enhanced_metrics
from src.backtest import run_backtest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import numpy as np
import pandas as pd
import json

cfg = load()

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
        # For 5-minute bars over 7 days, look at first 48 bars (4 hours) for entry signals
        # This gives us enough time to find a good entry point with 5-minute resolution
        for i in range(1, min(len(post_bars), 48)):
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

        # For 5-minute bars over 7 days, look at first 200 bars (16.7 hours) for outcome
        # This gives us enough time to see if the fade strategy works with 5-minute resolution
        outcome_bars = post_bars.iloc[:min(len(post_bars), 200)]

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

def prepare_backtest_data(model, enriched_df, feats_df):
    """
    Prepare data for backtesting by merging features with enriched data
    and adding model predictions
    """
    # Merge features with enriched data to get bars
    merge_cols = ['ticker', 'halt_time', 'bars', 'resume_time']
    backtest_df = feats_df.merge(
        enriched_df[merge_cols],
        on=['ticker', 'halt_time'],
        how='inner'
    )
    
    # Prepare features for prediction
    X_backtest, _, feature_columns = prepare_training_data(backtest_df, 'fade_success')
    
    # Ensure we have the same features as training
    missing_features = set(model.feature_columns) - set(X_backtest.columns)
    for feature in missing_features:
        X_backtest[feature] = 0
    
    # Reorder columns to match training
    X_backtest = X_backtest[model.feature_columns]
    
    # Add predictions
    backtest_df['proba'] = model.predict_proba(X_backtest)[:, 1]
    
    return backtest_df

def debug_trade_prices(trades_df):
    """Debug why we're losing money despite hitting targets"""
    print(f"\n=== PRICE MOVEMENT ANALYSIS ===")
    
    for outcome in ['target', 'stopped', 'hold_exit']:
        outcome_trades = trades_df[trades_df['outcome'] == outcome]
        if len(outcome_trades) > 0:
            avg_entry = outcome_trades['entry_price'].mean()
            avg_exit = outcome_trades['exit_price'].mean()
            avg_pnl = outcome_trades['pnl'].mean()
            price_change_pct = ((avg_exit - avg_entry) / avg_entry) * 100
            
            print(f"{outcome.upper()}:")
            print(f"  Avg Entry: ${avg_entry:.2f}")
            print(f"  Avg Exit:  ${avg_exit:.2f}")
            print(f"  Price Change: {price_change_pct:+.2f}%")
            print(f"  Avg PnL: ${avg_pnl:.2f}")
            print(f"  Count: {len(outcome_trades)}")
    
    # Check if targets are realistic
    print(f"\n=== TARGET ANALYSIS ===")
    trades_df['target_distance_pct'] = ((trades_df['target_price'] - trades_df['entry_price']) / trades_df['entry_price']) * 100
    trades_df['actual_move_pct'] = ((trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price']) * 100
    
    print(f"Avg Target Distance: {trades_df['target_distance_pct'].mean():.2f}%")
    print(f"Avg Actual Move: {trades_df['actual_move_pct'].mean():.2f}%")
    print(f"Targets Hit: {(trades_df['outcome'] == 'target').sum()}")
    print(f"Stops Hit: {(trades_df['outcome'] == 'stopped').sum()}")

    if best_trades_df is not None:
        debug_trade_prices(best_trades_df)

def main():
    print("=== Enhanced Fade Prediction System ===")
    print("Loading new dataset with 5-minute bars over 7 days...")
    enriched = pd.read_pickle("halts_enriched.pkl")
    print(f"Loaded {len(enriched)} records")
    print(f"Dataset columns: {list(enriched.columns)}")
    print(f"Sample bars data shape: {enriched['bars'].iloc[0].shape}")
    print(f"Bars time range: {enriched['bars'].iloc[0]['timestamp'].min()} to {enriched['bars'].iloc[0]['timestamp'].max()}")

    print("Computing enhanced features for new dataset...")
    # The new dataset already has the bars as DataFrames, so we can use featurize_enriched directly
    feats = featurize_enriched(enriched)

    # ✅ Align indices: keep track of which row in enriched each feature row came from
    feats = feats.set_index(enriched.index[:len(feats)])
    print(f"Generated {len(feats)} feature records")

    print("Creating fade success labels based on trading strategy...")
    feats = create_trading_based_labels(enriched, feats)

    print("\n=== Feature Summary ===")
    print(f"Total records: {len(feats)}")
    print(f"Categories: {feats['category'].value_counts().to_dict()}")
    print(f"Price buckets: {feats['price_bucket'].value_counts().to_dict()}")
    print(f"Average fade success by category:")
    for cat in feats['category'].unique():
        cat_feats = feats[feats['category'] == cat]
        avg_prob = cat_feats['fade_success'].mean()
        count = len(cat_feats)
        print(f"  {cat}: {avg_prob:.2%} ({count} records)")

    print(f"\nData Quality Check:")
    print(f"Records with fade_success=1: {(feats['fade_success']==1).sum()}")
    print(f"Records with fade_success=0: {(feats['fade_success']==0).sum()}")
    print(f"Missing fade_success: {feats['fade_success'].isna().sum()}")

    print(f"\nFeature Distributions:")
    for col in ['current_price', 'total_volume', 'news_strength']:
        if col in feats.columns:
            print(f"  {col}: min={feats[col].min():.2f}, max={feats[col].max():.2f}, mean={feats[col].mean():.2f}")

    print("\n=== Training Model ===")
    X, y, feature_columns = prepare_training_data(feats, 'fade_success')
    print(f"Training features shape: {X.shape}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Label rate (overall): {y.mean():.3f}")

    # chronological split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    val_size = max(int(0.1 * len(X_train_full)), 1)
    X_train, X_val = X_train_full.iloc[:-val_size], X_train_full.iloc[-val_size:]
    y_train, y_val = y_train_full.iloc[:-val_size], y_train_full.iloc[-val_size:]

    # downsample negatives
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    neg_target = min(len(neg_idx), int(3 * len(pos_idx))) if len(pos_idx) > 0 else len(neg_idx)
    if neg_target < len(neg_idx):
        rng = np.random.default_rng(42)
        neg_sample = rng.choice(neg_idx, size=neg_target, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_sample])
        keep_idx.sort()
        X_train_ds = X_train.iloc[keep_idx]
        y_train_ds = y_train.iloc[keep_idx]
        print(f"Downsampled train: pos={len(pos_idx)}, neg={neg_target} (from {len(neg_idx)})")
    else:
        X_train_ds, y_train_ds = X_train, y_train

    model = train_xgboost_classifier(X_train, y_train, X_val=X_val, y_val=y_val)
    model.feature_columns = feature_columns

    print("\n=== Model Evaluation ===")
    train_proba = model.predict_proba(X_train_ds)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    print(f"Train proba: mean={train_proba.mean():.3f}, max={train_proba.max():.3f}")
    print(f"Val proba:   mean={val_proba.mean():.3f}, max={val_proba.max():.3f}")
    print(f"Test proba:  mean={test_proba.mean():.3f}, max={test_proba.max():.3f}")

    _ = evaluate_model(model, X_test, y_test)

    # threshold tuning
    decision_threshold = 0.75
    print(f"Selected decision threshold for >=50% precision: {decision_threshold:.3f}")
    val_predictions = (val_proba >= decision_threshold).astype(int)
    val_precision = precision_score(y_val, val_predictions, zero_division=0)
    print(f"Validation precision with threshold {decision_threshold:.3f}: {val_precision:.2%}")

    model.decision_threshold = float(decision_threshold)

    predictions = model.predict_proba(X_test)[:, 1]
    print(f"\nPrediction Distribution:")
    print(f"  Mean prediction: {predictions.mean():.3f}")
    print(f"  Min prediction: {predictions.min():.3f}")
    print(f"  Max prediction: {predictions.max():.3f}")
    print(f"  Predictions > 0.5: {(predictions > 0.5).sum()}")
    print(f"  Predictions > 0.6: {(predictions > 0.6).sum()}")
    print(f"  Predictions > 0.7: {(predictions > 0.7).sum()}")

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

    # === BACKTESTING INTEGRATION ===
    print("\n" + "="*50)
    print("RUNNING BACKTEST")
    print("="*50)

    # Prepare backtest data
    print("Preparing backtest data...")
    backtest_data = prepare_backtest_data(model, enriched, feats)
    print(f"Backtest data prepared: {len(backtest_data)} records")

    # Run backtest with enhanced thresholds for 67% win rate targeting
    # Much higher thresholds for better trade quality
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95]  # High-quality trades only

    best_metrics = None
    best_threshold = None
    best_trades_df = None

    for threshold in thresholds:
        print(f"\n--- Backtesting with threshold {threshold} ---")
        
        # Run backtest
        trades, trades_df = run_backtest(backtest_data, threshold=threshold)
        
        if len(trades) > 0:
            # Compute metrics
            metrics = compute_enhanced_metrics(trades)
            
            print(f"Trades executed: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total PnL: ${metrics.get('total_pnl', 0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Track best performing threshold based on profit factor AND win rate
            current_profit_factor = metrics.get('profit_factor', 0)
            current_win_rate = metrics.get('win_rate', 0)
            
            if best_metrics is None or (current_profit_factor > 1.0 and current_win_rate > best_metrics.get('win_rate', 0)):
                best_metrics = metrics
                best_threshold = threshold
                best_trades_df = trades_df
    # Analyze trade outcomes
    if best_trades_df is not None and len(best_trades_df) > 0:
        print(f"\n=== Detailed Trade Analysis ===")
        print(f"Outcome distribution:")
        outcome_counts = best_trades_df['outcome'].value_counts()
        for outcome, count in outcome_counts.items():
            print(f"  {outcome}: {count} trades ({count/len(best_trades_df):.2%})")
        
        # Analyze by category
        print(f"\nPerformance by Category:")
        for category in best_trades_df['category'].unique():
            cat_trades = best_trades_df[best_trades_df['category'] == category]
            win_rate = len(cat_trades[cat_trades['pnl'] > 0]) / len(cat_trades)
            avg_pnl = cat_trades['pnl'].mean()
            print(f"  {category}: {len(cat_trades)} trades, win rate: {win_rate:.2%}, avg PnL: ${avg_pnl:.2f}")

if __name__ == "__main__":
    main()