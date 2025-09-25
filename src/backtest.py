import pandas as pd
import numpy as np
from .sizing import calc_shares, apply_slippage
from .metrics import compute_metrics
from .config import load

cfg = load()

def simulate_trade(row, threshold=0.35, risk_reward=2.0, stop_loss_pct=0.05):
    """
    Simulate a single trade given model prediction and ticker bars.

    Args:
        row (pd.Series): one row with prediction, probability, and ticker data
            required keys:
                - proba: predicted probability of fade_success
                - bars: intraday bars (DataFrame with 'timestamp','open','high','low','close')
                - resume_time: timestamp when trading resumed
                - current_price: float
        threshold (float): decision threshold for entering trades
        risk_reward (float): reward: risk ratio (e.g. 2.0 = target is 2x stop)
        stop_loss_pct (float): stop loss in % of entry price

    Returns:
        dict: trade outcome with pnl, entry/exit info
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

    entry_bar = post_bars.iloc[0]
    entry_price = entry_bar['close']

    stop_price = entry_price * (1 + stop_loss_pct) 
    target_price = entry_price * (1 - stop_loss_pct * risk_reward)

    exit_price = entry_price
    exit_time = entry_bar['timestamp']
    outcome = "breakeven"

    for _, bar in post_bars.iterrows():
        low, high, ts = bar['low'], bar['high'], bar['timestamp']
        if high >= stop_price:
            exit_price = stop_price
            exit_time = ts
            outcome = "stopped"
            break
        if low <= target_price:
            exit_price = target_price
            exit_time = ts
            outcome = "target"
            break
    else:
        last_bar = post_bars.iloc[-1]
        exit_price = last_bar['close']
        exit_time = last_bar['timestamp']
        outcome = "hold_exit"

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
    }


def should_trade(event_row, fade_probability, pre_bars):
    """
    Apply client's filtering criteria
    """
    if event_row.get('is_biotech', False):
        return False
    
    current_price = event_row.get('current_price', 0)
    if current_price < 3 or current_price > 50: 
        return False
    
    if fade_probability < 0.45:  # Reduced from 0.6 to 0.45
        return False
    
    # total_volume = event_row.get('total_volume', 0)
    # if total_volume < 100000:  # Reduced from 1M to 100K shares
    #     return False
    # if 'volume' in pre_bars.columns:
    #     premarket_volume = pre_bars['volume'].iloc[:30].sum() if len(pre_bars) >= 30 else pre_bars['volume'].sum()
    #     if premarket_volume < 1_000_000:
    #         return False
    
    news_strength = event_row.get('news_strength', 1.0)
    if news_strength < 0.2:
        return False
    
    category = event_row.get('category', 'UNKNOWN')
    if category in ['IPO_SPAC', 'NO_NEWS_PARABOLIC']:
        return True 
    elif category in ['GAP_DOWN', 'MULTIPLE_GAP_DOWN']:
        return fade_probability > 0.5 
    
    return True

def find_entry_point(post, event_row, pre_bars):
    vwap = (pre_bars['close'] * pre_bars['volume']).sum() / pre_bars['volume'].sum() if pre_bars['volume'].sum() > 0 else pre_bars['close'].iloc[-1]
    hod = pre_bars['high'].max()

    for i in range(1, min(len(post), 40)):
        window = post.iloc[:i+1]
        if len(window) < 2:
            continue
        current_close = window['close'].iloc[-1]
        current_high = window['high'].iloc[-1]
        prev_high = window['high'].iloc[-2]

        vwap_fail = current_close < vwap
        hod_rejection = current_high < hod
        lower_high = current_high < prev_high

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
    Execute the trading strategy with scaling rules
    """
    partial_pct = 0.4 
    covered_shares_first = int(shares * partial_pct)
    remaining_shares = shares - covered_shares_first
    
    entry_price_adj = apply_slippage(entry_price, cfg.get('slippage_per_share', 0.02), shares, 'short')
    
    first_cover_done = False
    first_cover_pnl = 0
    
    for idx in post.loc[entry_idx:].index:
        row = post.loc[idx]
        
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
    
    last = post.iloc[-1]
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

def run_backtest(preds_df, threshold=0.35, risk_reward=2.0, stop_loss_pct=0.05):
    """
    Run backtest across all predictions.

    Args:
        preds_df (pd.DataFrame): rows with ['ticker','resume_time','bars','current_price','proba',...]
        threshold (float): entry threshold
        risk_reward (float): RR ratio
        stop_loss_pct (float): stop % (0.05 = 5%)

    Returns:
        trades (list of dict), trades_df (pd.DataFrame)
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

    cumulative_returns = df['trade_return'].cumsum().fillna(method='ffill').fillna(0.0)
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

