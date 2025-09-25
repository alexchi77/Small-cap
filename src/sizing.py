def calc_shares(price, risk_dollars, stop_pct):
    if stop_pct <= 0:
        return 0
    max_shares = int(risk_dollars / (price * stop_pct))
    return max_shares

def apply_slippage(price, slippage_per_share, shares, direction='short'):
    if direction == 'short':
        return price + slippage_per_share
    else:
        return price - slippage_per_share
