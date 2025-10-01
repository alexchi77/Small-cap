import pandas as pd
import numpy as np
from datetime import timedelta
from io import StringIO
from .config import load
cfg = load()

def load_enriched_csv(filepath, max_records=None):
    """
    Load enriched CSV file with proper parsing of bars DataFrames from string format.
    The CSV has a complex structure where bars data spans multiple rows.
    
    Args:
        filepath: Path to the CSV file
        max_records: Optional limit on number of records to load (for testing)
    """
    print(f"Reading CSV file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    record_pattern = r'^(\d+),([^,]+),([^,]*),([^,]*),([^,]*),"'
    
    lines = content.split('\n')
    record_starts = []
    for i, line in enumerate(lines):
        if re.match(record_pattern, line):
            record_starts.append(i)
    
    print(f"Found {len(record_starts)} records")
    
    if max_records:
        record_starts = record_starts[:max_records]
        print(f"Loading first {len(record_starts)} records only")
    
    rows = []
    for i, start_line in enumerate(record_starts):
        if i % 100 == 0:
            print(f"Processing record {i+1}/{len(record_starts)}")
            
        end_line = record_starts[i + 1] if i + 1 < len(record_starts) else len(lines)
        
        record_lines = lines[start_line:end_line]
        
        row_data = parse_single_record(record_lines)
        if row_data:
            rows.append(row_data)
    
    print(f"Successfully parsed {len(rows)} records")
    
    df = pd.DataFrame(rows)
    
    if 'halt_time' in df.columns:
        df['halt_time'] = pd.to_datetime(df['halt_time'], errors='coerce')
    if 'resume_time' in df.columns:
        df['resume_time'] = pd.to_datetime(df['resume_time'], errors='coerce')
    
    return df

def parse_single_record(record_lines):
    """
    Parse a single record from its lines.
    """
    try:
        first_line = record_lines[0].strip()
        parts = first_line.split(',', 5)
        
        if len(parts) < 6:
            return None
            
        row_data = {
            'index': parts[0],
            'ticker': parts[1],
            'halt_time': parts[2],
            'resume_time': parts[3] if parts[3] else None,
            'reason': parts[4] if parts[4] else None
        }
        
        bars_start = -1
        bars_end = -1
        
        for i, line in enumerate(record_lines):
            if 'timestamp' in line and 'open' in line and 'close' in line:
                bars_start = i
            elif line.strip().startswith('[') and 'rows x' in line:
                bars_end = i
                rest_data = line.split(']', 1)[1].strip()
                if rest_data.startswith(','):
                    rest_data = rest_data[1:]
                if rest_data:
                    remaining_parts = rest_data.split(',')
                    if len(remaining_parts) >= 13:
                        data_parts = remaining_parts[1:]
                        row_data['news'] = data_parts[0] if data_parts[0] else '[]'
                        news_classified_str = data_parts[1] if data_parts[1] else '[]'
                        if news_classified_str == '[]':
                            row_data['news_classified'] = []
                        else:
                            try:
                                import ast
                                row_data['news_classified'] = ast.literal_eval(news_classified_str)
                            except:
                                row_data['news_classified'] = []
                        
                        try:
                            row_data['pre_halt_return'] = float(data_parts[2]) if data_parts[2] and data_parts[2] != '[]' else 0.0
                        except ValueError:
                            row_data['pre_halt_return'] = 0.0
                            
                        try:
                            row_data['pre_halt_vol_ratio'] = float(data_parts[3]) if data_parts[3] and data_parts[3] != '[]' else 0.0
                        except ValueError:
                            row_data['pre_halt_vol_ratio'] = 0.0
                            
                        try:
                            row_data['vwap'] = float(data_parts[4]) if data_parts[4] and data_parts[4] != '[]' else np.nan
                        except ValueError:
                            row_data['vwap'] = np.nan
                            
                        try:
                            row_data['gap_from_prev_close'] = float(data_parts[5]) if data_parts[5] and data_parts[5] != '[]' else 0.0
                        except ValueError:
                            row_data['gap_from_prev_close'] = 0.0
                        
                        row_data['category'] = data_parts[6] if data_parts[6] and data_parts[6] != '[]' else 'UNKNOWN'
                        row_data['news_flag'] = data_parts[7] == 'True' if data_parts[7] else False
                        
                        try:
                            row_data['parabolic_pct'] = float(data_parts[8]) if data_parts[8] and data_parts[8] != '[]' else 0.0
                        except ValueError:
                            row_data['parabolic_pct'] = 0.0
                            
                        row_data['ipo_flag'] = data_parts[9] == 'True' if data_parts[9] else False
                        
                        try:
                            row_data['news_sentiment'] = float(data_parts[10]) if data_parts[10] and data_parts[10] != '[]' else 0.0
                        except ValueError:
                            row_data['news_sentiment'] = 0.0
                            
                        try:
                            row_data['fade_success'] = int(data_parts[11].strip()) if data_parts[11] and data_parts[11] != '[]' else 0
                        except ValueError:
                            row_data['fade_success'] = 0
                break
        
        if bars_start >= 0 and bars_end >= 0:
            bars_lines = record_lines[bars_start:bars_end]
            row_data['bars'] = parse_bars_from_lines(bars_lines)
        else:
            row_data['bars'] = pd.DataFrame()
        
        return row_data
        
    except Exception as e:
        print(f"Error parsing record: {e}")
        return None

def parse_bars_from_lines(bars_lines):
    """
    Parse bars data from the list of lines containing the DataFrame representation.
    """
    if not bars_lines:
        return pd.DataFrame()
    
    try:
        header_idx = -1
        for i, line in enumerate(bars_lines):
            if 'timestamp' in line and 'open' in line and 'close' in line:
                header_idx = i
                break
        
        if header_idx == -1:
            return pd.DataFrame()
        
        data_lines = bars_lines[header_idx + 1:]
        
        data_lines = [line.strip() for line in data_lines 
                     if line.strip() and not line.strip().startswith('..')]
        
        if not data_lines:
            return pd.DataFrame()
        
        rows = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    row_data = {
                        'timestamp': pd.to_datetime(parts[1] + ' ' + parts[2]),
                        'open': float(parts[3]),
                        'high': float(parts[4]),
                        'low': float(parts[5]),
                        'close': float(parts[6]),
                        'volume': float(parts[7])
                    }
                    rows.append(row_data)
                except (ValueError, IndexError):
                    continue
        
        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error parsing bars data: {e}")
        return pd.DataFrame()

def compute_vwap(bars):
    if bars.empty: return np.nan
    typical = (bars['close'] * bars['volume']).cumsum()
    vwap = typical.iloc[-1] / bars['volume'].cumsum().iloc[-1]
    return vwap

def detect_parabolic(bars, pct_threshold=0.25, window_minutes=15):
    if bars.empty: return False, 0.0
    end = bars['timestamp'].max()
    start = end - pd.Timedelta(minutes=window_minutes)
    sel = bars[(bars['timestamp'] >= start) & (bars['timestamp'] <= end)]
    if sel.empty or len(sel)<2: return False, 0.0
    pct = (sel['close'].iloc[-1] / sel['close'].iloc[0]) - 1.0
    return pct >= pct_threshold, pct

def classify_halt(row):
    """
    Enhanced halt classification based on client's strategy requirements
    """
    bars = row['bars'] if isinstance(row['bars'], pd.DataFrame) else pd.DataFrame()
    
    news_classified_raw = row.get('news_classified', [])
    if isinstance(news_classified_raw, str):
        if news_classified_raw == '[]' or news_classified_raw == '':
            news_classified = []
        else:
            try:
                import ast
                news_classified = ast.literal_eval(news_classified_raw)
            except:
                news_classified = []
    elif isinstance(news_classified_raw, (int, float)):
        news_classified = []
    else:
        news_classified = news_classified_raw if news_classified_raw is not None else []
    
    news_flag = len(news_classified) > 0
    is_para, pct = detect_parabolic(bars, pct_threshold=0.25, window_minutes=15)
    
    ipo_flag = any(isinstance(n, dict) and n.get('category') == 'IPO_SPAC' for n in news_classified) or ('ipo' in str(row.get('reason','')).lower()) or ('spac' in str(row.get('reason','')).lower())
    
    is_gap_down = detect_gap_down(bars)
    
    multiple_gap_downs = detect_multiple_gap_downs(bars)
    
    is_biotech = detect_biotech(row)
    
    if ipo_flag:
        cat = "IPO_SPAC"
    elif is_gap_down and multiple_gap_downs:
        cat = "MULTIPLE_GAP_DOWN"
    elif is_gap_down:
        cat = "GAP_DOWN"
    elif is_para and not news_flag:
        cat = "NO_NEWS_PARABOLIC"
    elif news_flag:
        cat = "NEWS_DRIVEN"
    else:
        cat = "UNKNOWN"
    
    return {
        "category": cat,
        "news_flag": news_flag,
        "parabolic_pct": pct,
        "ipo_flag": ipo_flag,
        "is_gap_down": is_gap_down,
        "multiple_gap_downs": multiple_gap_downs,
        "is_biotech": is_biotech,
        "news_sentiment": (
                            sum([n.get('sentiment', 0.0) for n in news_classified if isinstance(n, dict)])
                            / max(1, len([n for n in news_classified if isinstance(n, dict)]))
                        ) if news_classified else 0.0
    }

def detect_gap_down(bars):
    """Detect if there's a significant gap down"""
    if bars.empty or len(bars) < 2:
        return False
    
    pre_close = bars['close'].iloc[0]
    current_price = bars['close'].iloc[-1]
    gap_pct = (current_price - pre_close) / pre_close if pre_close > 0 else 0
    
    return gap_pct < -0.05 

def detect_multiple_gap_downs(bars):
    """Detect if there are multiple gap downs intraday"""
    if bars.empty or len(bars) < 10:
        return False
    
    gap_downs = 0
    for i in range(1, len(bars)):
        prev_close = bars['close'].iloc[i-1]
        current_open = bars['open'].iloc[i]
        gap_pct = (current_open - prev_close) / prev_close if prev_close > 0 else 0
        
        if gap_pct < -0.03:
            gap_downs += 1
    
    return gap_downs >= 2

def detect_biotech(row):
    """Detect if the stock is likely biotech (for filtering)"""
    # print(row.get('ticker', ''))
    ticker = str(row.get("ticker", "") or "").upper()
    # desc   = str(row.get("description", "") or "").lower()
    reason = str(row.get('reason', '')).lower()
    
    biotech_keywords = ['fda', 'clinical', 'trial', 'drug', 'pharma', 'biotech', 'therapeutic']
    
    if any(keyword in reason for keyword in biotech_keywords):
        return True
    
    return False

def featurize_enriched(enriched_df):
    rows = []
    for _,r in enriched_df.iterrows():
        bars = r['bars'] if isinstance(r['bars'], pd.DataFrame) else pd.DataFrame()
        if bars.empty:
            feats = {"pre_halt_return": 0.0, "pre_halt_vol_ratio": 0.0, "vwap": np.nan, "gap_from_prev_close": 0.0}
        else:
            end = bars['timestamp'].max()
            start = end - pd.Timedelta(minutes=30)
            sel = bars[(bars['timestamp']>=start)&(bars['timestamp']<=end)]
            if sel.empty:
                pre_ret = 0.0
                pre_vol = bars['volume'].iloc[-1] if len(bars)>0 else 0.0
            else:
                pre_ret = sel['close'].iloc[-1] / sel['close'].iloc[0] - 1.0
                pre_vol = sel['volume'].mean()
            avg_vol = bars['volume'].mean() if not bars.empty else 1.0
            vol_ratio = pre_vol / (avg_vol + 1e-9)
            vwap = compute_vwap(bars)
            prev_close = bars['close'].iloc[0] if len(bars)>0 else np.nan
            gap = (bars['close'].iloc[-1] / prev_close - 1.0) if not pd.isna(prev_close) and prev_close!=0 else 0.0
            
            enhanced_feats = compute_enhanced_features(bars, r)
            
            feats = {
                "pre_halt_return": pre_ret, 
                "pre_halt_vol_ratio": vol_ratio, 
                "vwap": vwap, 
                "gap_from_prev_close": gap,
                **enhanced_feats
            }
        
        cls = classify_halt(r)
        out = {**feats, **cls}
        out.update({"ticker": r['ticker'], "halt_time": r['halt_time'], "resume_time": r['resume_time']})
        rows.append(out)
    return pd.DataFrame(rows)

def compute_enhanced_features(bars, row):
    """
    Enhanced features for 67% win rate targeting
    """
    if bars.empty:
        return {}
    
    features = {}
    
    current_price = bars['close'].iloc[-1] if len(bars) > 0 else 0
    features['current_price'] = current_price
    features['price_bucket'] = get_price_bucket(current_price)
    
    # Enhanced volume analysis
    total_volume = bars['volume'].sum()
    features['total_volume'] = total_volume
    features['avg_volume'] = bars['volume'].mean()
    features['volume_std'] = bars['volume'].std() if len(bars) > 1 else 0
    
    # Volume momentum and acceleration
    if len(bars) >= 10:
        recent_vol = bars['volume'].iloc[-5:].mean()
        early_vol = bars['volume'].iloc[:5].mean()
        features['volume_acceleration'] = recent_vol / (early_vol + 1e-9)
    else:
        features['volume_acceleration'] = 1.0
    
    # Volume percentile and trend
    features['volume_percentile'] = bars['volume'].rank(pct=True).iloc[-1]
    if len(bars) > 1:
        features['volume_trend'] = np.polyfit(range(len(bars)), bars['volume'], 1)[0]
    else:
        features['volume_trend'] = 0
    
    halt_time = pd.to_datetime(row['halt_time'])
    features['halt_hour'] = halt_time.hour
    features['halt_minute'] = halt_time.minute
    features['time_of_day'] = get_time_of_day_category(halt_time)
    
    # Enhanced VWAP analysis
    vwap = compute_vwap(bars)
    features['distance_from_vwap'] = (current_price - vwap) / vwap if vwap > 0 else 0
    features['vwap_extended'] = abs(features['distance_from_vwap']) > 0.05 
    
    # VWAP momentum
    if len(bars) >= 20:
        vwap_20 = compute_vwap(bars.iloc[-20:])
        features['vwap_momentum'] = (vwap - vwap_20) / vwap_20 if vwap_20 > 0 else 0
    else:
        features['vwap_momentum'] = 0
    
    pre_halt_close = bars['close'].iloc[0] if len(bars) > 0 else current_price
    halt_open = bars['open'].iloc[-1] if len(bars) > 0 else current_price
    features['gap_size'] = (halt_open - pre_halt_close) / pre_halt_close if pre_halt_close > 0 else 0
    features['gap_direction'] = 'up' if features['gap_size'] > 0.02 else 'down' if features['gap_size'] < -0.02 else 'neutral'
    
    # Enhanced gap analysis
    features['gap_momentum'] = 0
    gap_count = 0
    for i in range(1, len(bars)):
        prev_close = bars['close'].iloc[i-1]
        current_open = bars['open'].iloc[i]
        gap_pct = (current_open - prev_close) / prev_close if prev_close > 0 else 0
        if gap_pct < -0.02:
            gap_count += 1
            features['gap_momentum'] += gap_pct
    features['gap_count'] = gap_count
    
    features['multiple_halts'] = detect_multiple_halts(bars)
    
    # Enhanced support/resistance analysis
    features['hod'] = bars['high'].max()
    features['lod'] = bars['low'].min()
    features['hod_rejection'] = current_price < features['hod'] * 0.98 
    features['weakness_confirmed'] = features['hod_rejection'] and current_price < vwap
    
    # Distance to key levels
    features['distance_to_hod'] = (features['hod'] - current_price) / current_price
    features['distance_to_lod'] = (current_price - features['lod']) / current_price
    
    # Technical indicators
    if len(bars) >= 20:
        # RSI calculation
        price_changes = bars['close'].diff().dropna()
        gains = price_changes[price_changes > 0].sum()
        losses = abs(price_changes[price_changes < 0].sum())
        rs = gains / (losses + 1e-9)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = bars['close'].rolling(window=20).mean().iloc[-1]
        std = bars['close'].rolling(window=20).std().iloc[-1]
        features['bb_position'] = (current_price - sma) / (2 * std + 1e-9)
    else:
        features['rsi'] = 50
        features['bb_position'] = 0
    
    # Momentum analysis
    if len(bars) >= 10:
        features['short_momentum'] = (bars['close'].iloc[-3:].mean() / bars['close'].iloc[-6:-3].mean()) - 1.0
        features['long_momentum'] = (bars['close'].iloc[-10:].mean() / bars['close'].iloc[:10].mean()) - 1.0
    else:
        features['short_momentum'] = 0
        features['long_momentum'] = 0
    
    # Volume-price relationship
    features['volume_price_correlation'] = bars['volume'].corr(bars['close'])
    
    # Market structure
    higher_highs = 0
    lower_lows = 0
    for i in range(2, len(bars)):
        if bars['high'].iloc[i] > bars['high'].iloc[i-1] > bars['high'].iloc[i-2]:
            higher_highs += 1
        if bars['low'].iloc[i] < bars['low'].iloc[i-1] < bars['low'].iloc[i-2]:
            lower_lows += 1
    features['market_structure'] = higher_highs - lower_lows
    
    features['news_strength'] = compute_news_strength(row.get('news_classified', []))
    
    features['second_day_continuation'] = 0 
    features['float_bucket'] = 'unknown' 
    
    return features

def get_price_bucket(price):
    """Categorize price into buckets based on client's performance data"""
    if 5 <= price < 10:
        return '5_10'
    elif 10 <= price < 20:
        return '10_20'
    elif 20 <= price < 50:
        return '20_50'
    elif 50 <= price < 100:
        return '50_100'
    else:
        return 'other'

def get_time_of_day_category(halt_time):
    """Categorize halt time based on client's requirements"""
    hour = halt_time.hour
    if 9 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 15:
        return 'midday'
    else:
        return 'late_day'

def detect_multiple_halts(bars):
    """Detect if there are signs of multiple halts (simplified)"""
    return 0

def compute_news_strength(news_classified):
    """Compute news strength score based on client's requirements"""
    if not news_classified:
        return 1.0 
    
    strength_map = {
        'FDA_APPROVAL': 0.1, 
        'EARNINGS': 0.3,
        'BUYOUT': 0.1,
        'SMALL_CONTRACT': 0.8,
        'PR_FLUFF': 0.9,
        'IPO_SPAC': 0.4
    }
    
    max_strength = 1.0 
    for news in news_classified:
        if isinstance(news, dict):
            category = news.get('category', '')
            strength = strength_map.get(category, 0.5)
            max_strength = min(max_strength, strength)
    
    return max_strength

def merge_features_into_enriched(enriched_df, feats_df):
    """
    Merge features back into enriched DataFrame, handling potential duplicates
    """
    enriched_copy = enriched_df.copy()
    feats_df = feats_df.copy()
    
    enriched_duplicates = enriched_copy.duplicated(subset=['ticker', 'halt_time']).sum()
    feats_duplicates = feats_df.duplicated(subset=['ticker', 'halt_time']).sum()
    
    if enriched_duplicates > 0:
        print(f"Warning: Found {enriched_duplicates} duplicate (ticker, halt_time) combinations in enriched data")
        enriched_copy = enriched_copy.drop_duplicates(subset=['ticker', 'halt_time'], keep='first')
    
    if feats_duplicates > 0:
        print(f"Warning: Found {feats_duplicates} duplicate (ticker, halt_time) combinations in features data")
        feats_df = feats_df.drop_duplicates(subset=['ticker', 'halt_time'], keep='first')
    
    merged_df = enriched_copy.merge(
        feats_df, 
        on=['ticker', 'halt_time'], 
        how='left',
        suffixes=('', '_feat')
    )
    
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    return merged_df
