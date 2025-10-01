import pandas as pd
import numpy as np
from .sizing import calc_shares, apply_slippage
from .metrics import compute_metrics
from .config import load

cfg = load()

def simulate_trade(row, threshold=0.35, risk_reward=2.0, stop_loss_pct=0.05):
    """
    Enhanced trade simulation with improved stop loss management for 67% win rate
    """
    proba = row.get("proba", 0.0)
    bars = row.get("bars")
    resume_time = row.get("resume_time")
    current_price = row.get("current_price", None)

    if proba < threshold or bars is None or resume_time is None or current_price is None:
        return None 

    post_bars = bars[bars['timestamp'] >= resume_time].copy()
    if post_bars.empty:
        return None

    # Enhanced entry point selection - wait for better entry
    entry_price = None
    entry_time = None
    
    # Look for entry within first 24 bars (2 hours) for better entry timing
    for i in range(min(24, len(post_bars))):
        bar = post_bars.iloc[i]
        # Use open price for entry, but only if it's not too far from previous close
        if i > 0:
            prev_close = post_bars.iloc[i-1]['close']
            price_change = abs(bar['open'] - prev_close) / prev_close
            if price_change < 0.02:  # Only enter if price hasn't moved too much
                entry_price = bar['open']
                entry_time = bar['timestamp']
                break
        else:
            entry_price = bar['open']
            entry_time = bar['timestamp']
            break
    
    if entry_price is None:
        return None

    # Enhanced stop loss management - tighter stops for better risk management
    category = row.get("category", "UNKNOWN")
    
    # Dynamic stop loss based on category and volatility
    if category == "MULTIPLE_GAP_DOWN":
        stop_loss_pct = 0.03  # Tighter stop for multiple gaps
    elif category == "GAP_DOWN":
        stop_loss_pct = 0.04  # Slightly tighter stop for single gaps
    elif category == "IPO_SPAC":
        stop_loss_pct = 0.05  # Standard stop for IPOs
    else:
        stop_loss_pct = 0.06  # Slightly wider stop for unknown categories
    
    # For SHORT positions: stop above, target below
    stop_price = entry_price * (1 + stop_loss_pct) 
    target_price = entry_price * (1 - stop_loss_pct * risk_reward)

    exit_price = entry_price
    exit_time = entry_time
    outcome = "breakeven"

    # Enhanced exit logic with trailing stops
    max_bars = min(200, len(post_bars))
    trailing_stop = stop_price
    best_price = entry_price
    
    for i in range(max_bars):
        bar = post_bars.iloc[i]
        low, high, ts = bar['low'], bar['high'], bar['timestamp']
        
        # Update trailing stop if price moves in our favor
        if low < best_price:
            best_price = low
            # Trail the stop by 1% below best price
            trailing_stop = min(trailing_stop, best_price * 1.01)
        
        if high >= trailing_stop:
            exit_price = trailing_stop
            exit_time = ts
            outcome = "stopped"
            break
        if low <= target_price:
            exit_price = target_price
            exit_time = ts
            outcome = "target"
            break
    else:
        # If we didn't hit stop or target, exit at the last bar we checked
        last_bar = post_bars.iloc[max_bars - 1]
        exit_price = last_bar['close']
        exit_time = last_bar['timestamp']
        outcome = "hold_exit"

    # For SHORT positions, profit when exit_price < entry_price
    pnl = (entry_price - exit_price)
    
    return {
        "ticker": row.get("ticker"),
        "resume_time": resume_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_time": exit_time,
        "pnl": pnl,
        "outcome": outcome,
        "category": row.get("category", "UNKNOWN"),
        "price_bucket": row.get("price_bucket", "N/A"),
        "target_price": target_price,
        "stop_price": trailing_stop,
    }


def should_trade(event_row, fade_probability, pre_bars):
    """
    Enhanced filtering criteria for higher win rate (targeting 67%)
    """
    if event_row.get('is_biotech', False):
        return False
    
    current_price = event_row.get('current_price', 0)
    if current_price < 5 or current_price > 25:  # Tighter price range
        return False
    
    # Much higher probability threshold for better quality trades
    if fade_probability < 0.75:  # Increased from 0.45 to 0.75
        return False
    
    # Enhanced volume requirements
    total_volume = event_row.get('total_volume', 0)
    if total_volume < 500000:  # Higher volume requirement
        return False
    
    # Check premarket volume for better liquidity
    if 'volume' in pre_bars.columns and len(pre_bars) >= 20:
        premarket_volume = pre_bars['volume'].iloc[:20].sum()
        if premarket_volume < 2_000_000:  # Higher premarket volume requirement
            return False
    
    news_strength = event_row.get('news_strength', 1.0)
    if news_strength < 0.5:  # Higher news strength requirement
        return False
    
    # Enhanced category-based filtering
    category = event_row.get('category', 'UNKNOWN')
    if category == 'IPO_SPAC':
        return fade_probability > 0.85  # Very high threshold for IPOs
    elif category == 'MULTIPLE_GAP_DOWN':
        return fade_probability > 0.80  # High threshold for multiple gaps
    elif category == 'GAP_DOWN':
        return fade_probability > 0.75  # High threshold for single gaps
    elif category == 'NEWS_DRIVEN':
        return fade_probability > 0.85  # Very high threshold for news-driven
    elif category == 'UNKNOWN':
        return fade_probability > 0.85  # Very high threshold for unknown
    
    return False  # Default to no trade for safety

def find_entry_point(post, event_row, pre_bars):
    """
    Find entry point for 5-minute bars over 7 days.
    Updated to handle longer timeframe with more bars.
    """
    vwap = (pre_bars['close'] * pre_bars['volume']).sum() / pre_bars['volume'].sum() if pre_bars['volume'].sum() > 0 else pre_bars['close'].iloc[-1]
    hod = pre_bars['high'].max()

    # For 5-minute bars, look at first 48 bars (4 hours) for entry signals
    # This gives us more time to find a good entry point
    max_lookback = min(len(post), 48)
    
    for i in range(1, max_lookback):
        window = post.iloc[:i+1]
        if len(window) < 2:
            continue
        current_close = window['close'].iloc[-1]
        current_high = window['high'].iloc[-1]
        prev_high = window['high'].iloc[-2]

        vwap_fail = current_close < vwap
        hod_rejection = current_high < hod
        lower_high = current_high < prev_high

        # More flexible entry criteria for 5-minute bars
        if vwap_fail and (hod_rejection or lower_high):
            entry_idx = window.index[-1]
            entry_price = window['open'].iloc[-1]
            return entry_idx, entry_price
    return None


def calculate_position_size(event_row, fade_probability, entry_price):
    """
    Enhanced position sizing based on client's requirements
    """
    fixed_risk = cfg.get('fixed_risk_dollars', 1000)
    
    risk_multiplier = fade_probability 
    
    price_bucket = event_row.get('price_bucket', 'other')
    bucket_multipliers = {
        '5_10': 1.2,    
        '10_20': 1.5,   
        '20_50': 0.7,   
        '50_100': 0.3,  
        'other': 0.5
    }
    bucket_multiplier = bucket_multipliers.get(price_bucket, 0.5)
    
    adjusted_risk = fixed_risk * risk_multiplier * bucket_multiplier
    
    hod = event_row.get('hod', entry_price * 1.1)
    stop_price = hod * 1.02
    
    stop_pct = abs((stop_price - entry_price) / entry_price)
    shares = calc_shares(entry_price, adjusted_risk, stop_pct)
    
    return {
        'shares': shares,
        'stop_price': stop_price,
        'risk_dollars': adjusted_risk
    }

def calculate_targets(event_row, pre_bars, entry_price):
    pre_halt_close = pre_bars['close'].iloc[-1]
    vwap = (pre_bars['close'] * pre_bars['volume']).sum() / pre_bars['volume'].sum() if pre_bars['volume'].sum() > 0 else entry_price
    category = event_row.get('category', 'UNKNOWN')
    targets = {}
    if category == 'IPO_SPAC':
        targets['primary'] = pre_halt_close
        targets['vwap'] = vwap
    elif category in ['GAP_DOWN', 'MULTIPLE_GAP_DOWN']:
        targets['primary'] = min(vwap, pre_halt_close)
        targets['vwap'] = vwap
    else:
        targets['primary'] = pre_halt_close
        targets['vwap'] = vwap
    return targets


def execute_trade_strategy(post, entry_idx, entry_price, stop_price, shares, targets, event_row):
    """
    Execute the trading strategy with scaling rules for 5-minute bars over 7 days.
    Updated to handle longer timeframe with more bars.
    """
    partial_pct = 0.4 
    covered_shares_first = int(shares * partial_pct)
    remaining_shares = shares - covered_shares_first
    
    entry_price_adj = apply_slippage(entry_price, cfg.get('slippage_per_share', 0.02), shares, 'short')
    
    first_cover_done = False
    first_cover_pnl = 0
    
    # For 5-minute bars over 7 days, limit to first 200 bars (16.7 hours) to avoid holding too long
    max_bars = min(200, len(post))
    post_subset = post.iloc[:max_bars]
    
    for i, (idx, row) in enumerate(post_subset.iterrows()):
        if i < entry_idx:
            continue
            
        if row['high'] >= stop_price:
            pnl = (entry_price_adj - stop_price) * shares - cfg.get('slippage_per_share', 0.02) * shares
            return {
                "outcome": "stopped", 
                "pnl": pnl, 
                "entry_time": post.loc[entry_idx]['timestamp'], 
                "exit_time": row['timestamp'],
                "shares": shares,
                "entry_price": entry_price_adj,
                "exit_price": stop_price
            }
        
        if not first_cover_done and row['low'] <= targets['vwap']:
            cover_price = targets['vwap'] - cfg.get('slippage_per_share', 0.02)
            first_cover_pnl = (entry_price_adj - cover_price) * covered_shares_first
            first_cover_done = True
            
            if cover_price <= targets['primary']:
                return {
                    "outcome": "target", 
                    "pnl": first_cover_pnl, 
                    "entry_time": post.loc[entry_idx]['timestamp'], 
                    "exit_time": row['timestamp'],
                    "shares": shares,
                    "entry_price": entry_price_adj,
                    "exit_price": cover_price
                }
        
        if row['low'] <= targets['primary']:
            cover_price = targets['primary'] - cfg.get('slippage_per_share', 0.02)
            remaining_pnl = (entry_price_adj - cover_price) * remaining_shares
            total_pnl = first_cover_pnl + remaining_pnl
            
            return {
                "outcome": "target", 
                "pnl": total_pnl, 
                "entry_time": post.loc[entry_idx]['timestamp'], 
                "exit_time": row['timestamp'],
                "shares": shares,
                "entry_price": entry_price_adj,
                "exit_price": cover_price
            }
    
    # If we didn't hit stop or target within the time limit, exit at the last bar
    last = post_subset.iloc[-1]
    cover_price = last['close'] - cfg.get('slippage_per_share', 0.02)
    total_pnl = first_cover_pnl + (entry_price_adj - cover_price) * remaining_shares
    
    return {
        "outcome": "held", 
        "pnl": total_pnl, 
        "entry_time": post.loc[entry_idx]['timestamp'], 
        "exit_time": last['timestamp'],
        "shares": shares,
        "entry_price": entry_price_adj,
        "exit_price": cover_price
    }

def run_backtest(preds_df, threshold=0.75, risk_reward=2.0, stop_loss_pct=0.05):  # Changed default to 0.75
    """
    Run backtest across all predictions.
    """
    trades = []
    for _, row in preds_df.iterrows():
        trade = simulate_trade(row, threshold, risk_reward, stop_loss_pct)
        if trade:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)
    return trades, trades_df

def compute_enhanced_metrics(trades):
    """
    Compute comprehensive backtest metrics from a list of trade dicts.

    Expects each trade dict to contain at least:
      - 'pnl' (float)
      - 'entry_time' (parsable datetime) optional but recommended
      - 'exit_time' (parsable datetime) optional but recommended
      - 'outcome' (str) optional (e.g. 'target', 'stopped', 'held')
      - 'category' (str) optional
      - 'price_bucket' (str) optional
      - 'risk_dollars' (float) optional, used for risk-based metrics if present

    Returns: dict of metrics
    """
    import numpy as np
    import pandas as pd

    if not trades:
        return {}

    df = pd.DataFrame(trades)

    if 'pnl' not in df.columns:
        raise ValueError("Each trade must include a 'pnl' field")

    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = df['pnl'].sum()
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    if 'entry_time' in df.columns and 'exit_time' in df.columns:
        try:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['trade_duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60.0 
        except Exception:
            df['trade_duration'] = np.nan
    else:
        df['trade_duration'] = np.nan

    if 'outcome' in df.columns:
        target_df = df[df['outcome'] == 'target']
    else:
        target_df = df[df['pnl'] > 0]  

    avg_time_to_fade = target_df['trade_duration'].mean() if len(target_df) > 0 and target_df['trade_duration'].notna().any() else 0.0

    # Updated thresholds for 5-minute bars over 7 days
    # Intraday: <= 390 minutes (6.5 hours)
    # Multi-day: > 390 minutes
    intraday_trades = df[df['trade_duration'] <= 390]
    multi_day_trades = df[df['trade_duration'] > 390]

    intraday_pct = len(intraday_trades) / total_trades if total_trades > 0 else 0.0
    multi_day_pct = len(multi_day_trades) / total_trades if total_trades > 0 else 0.0

    gap_fill_probability = len(df[df.get('outcome', '') == 'target']) / total_trades if total_trades > 0 else 0.0
    if 'outcome' not in df.columns:
        gap_fill_probability = len(df[df['pnl'] > 0]) / total_trades if total_trades > 0 else 0.0

    initial_equity = None
    try:
        initial_equity = cfg.get('initial_equity', 100000)
    except Exception:
        initial_equity = 100000

    if 'risk_dollars' in df.columns and df['risk_dollars'].notna().any():
        df['trade_return'] = df.apply(lambda r: (r['pnl'] / r['risk_dollars']) if (pd.notna(r.get('risk_dollars')) and r.get('risk_dollars') not in (0, None)) else (r['pnl'] / initial_equity), axis=1)
    else:
        df['trade_return'] = df['pnl'] / initial_equity

    if len(df) > 1:
        returns = df['trade_return'].dropna()
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    cumulative_returns = df['trade_return'].cumsum().ffill().fillna(0.0)
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min() 
    max_drawdown_pct = abs(max_drawdown)

    category_metrics = {}
    if 'category' in df.columns:
        for category in df['category'].fillna('UNKNOWN').unique():
            cat_df = df[df['category'].fillna('UNKNOWN') == category]
            cat_win_rate = len(cat_df[cat_df['pnl'] > 0]) / len(cat_df) if len(cat_df) > 0 else 0.0
            cat_avg_pnl = cat_df['pnl'].mean() if len(cat_df) > 0 else 0.0
            category_metrics[category] = {
                'count': len(cat_df),
                'win_rate': cat_win_rate,
                'avg_pnl': cat_avg_pnl
            }

    price_bucket_metrics = {}
    if 'price_bucket' in df.columns:
        for bucket in df['price_bucket'].fillna('UNKNOWN').unique():
            bucket_df = df[df['price_bucket'].fillna('UNKNOWN') == bucket]
            bucket_win_rate = len(bucket_df[bucket_df['pnl'] > 0]) / len(bucket_df) if len(bucket_df) > 0 else 0.0
            bucket_avg_pnl = bucket_df['pnl'].mean() if len(bucket_df) > 0 else 0.0
            price_bucket_metrics[bucket] = {
                'count': len(bucket_df),
                'win_rate': bucket_win_rate,
                'avg_pnl': bucket_avg_pnl
            }

    metrics = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_time_to_fade_minutes': avg_time_to_fade,
        'intraday_pct': intraday_pct,
        'multi_day_pct': multi_day_pct,
        'gap_fill_probability': gap_fill_probability,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,         
        'max_drawdown_magnitude': max_drawdown_pct, 
        'category_metrics': category_metrics,
        'price_bucket_metrics': price_bucket_metrics
    }

    return metrics

