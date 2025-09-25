import numpy as np
import pandas as pd

def compute_metrics(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return {}
    dfp = df[df['pnl'].notnull()]
    wins = dfp[dfp['pnl']>0]
    losses = dfp[dfp['pnl']<=0]
    win_pct = len(wins)/len(dfp) if len(dfp)>0 else np.nan
    avg_pnl = dfp['pnl'].mean()
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if losses['pnl'].sum()!=0 else np.inf
    sharpe = dfp['pnl'].mean() / (dfp['pnl'].std()+1e-9)
    cr = dfp['pnl'].cumsum()
    if cr.empty:
        maxdd = 0.0
    else:
        peak = cr.cummax()
        dd = cr - peak
        maxdd = dd.min()
    if 'halt_time' in df.columns:
        df['halt_time'] = pd.to_datetime(df['halt_time'])
        period_days = (df['halt_time'].max() - df['halt_time'].min()).days or 1
        freq_per_year = len(df)/period_days * 365
    else:
        freq_per_year = np.nan
    return {
        "n_trades": len(dfp),
        "win_pct": win_pct,
        "avg_pnl": avg_pnl,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_drawdown": maxdd,
        "freq_per_year": freq_per_year
    }
