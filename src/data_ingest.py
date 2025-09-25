import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from .config import load
from .news_openai import classify_news_items
from .utils import ensure_dir

cfg = load()
POLY_KEY = cfg["polygon_api_key"]
BASE = "https://api.polygon.io"

def load_local_halts(path=None):
    path = path or cfg["data_path"]
    df = pd.read_csv(path, parse_dates=["halt_time", "resume_time"], low_memory=False)
    required = ["Symbol","halt_time","resume_time"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c} in local halts file.")
    return df

def fetch_bars_polygon(ticker, start_dt, end_dt, timespan="minute", limit=50000):
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"
    params = {"adjusted": "true", "sort": "asc", "limit": limit, "apiKey": POLY_KEY}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        time.sleep(0.5)
        r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "results" not in data:
        return pd.DataFrame()
    bars = pd.DataFrame(data["results"])
    bars['timestamp'] = pd.to_datetime(bars['t'], unit='ms')
    bars = bars.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return bars[['timestamp','open','high','low','close','volume']]

def fetch_news_polygon(ticker, start_dt, end_dt):
    url = f"{BASE}/v2/reference/news"
    params = {"ticker": ticker, "published_utc.gte": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "published_utc.lte": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "apiKey": POLY_KEY}
    # print(params)
    r = requests.get(url, params=params, timeout=30)
    # print(r)
    if r.status_code != 200:
        return []
    data = r.json()
    print(data)
    return data.get("results", [])

def enrich_single_halt(row, pre_minutes=60, post_minutes=60, timespan="minute"):
    t0 = pd.to_datetime(row['halt_time'])
    start = (t0 - pd.Timedelta(minutes=pre_minutes)).tz_localize(None)
    end = (t0 + pd.Timedelta(minutes=post_minutes)).tz_localize(None)
    try:
        bars = fetch_bars_polygon(row['Symbol'], start, end, timespan=timespan)
    except Exception as e:
        print("Polygon bars error:", e)
        bars = pd.DataFrame()
    try:
        news_items = fetch_news_polygon(row['Symbol'], start - pd.Timedelta(minutes=30), end + pd.Timedelta(minutes=0))
        print(news_items)
    except Exception as e:
        print("Polygon news error:", e)
        news_items = []
    classified = []
    if cfg.get('news',{}).get('use_openai', False) and len(news_items)>0:
        try:
            classified = classify_news_items(news_items)
        except Exception as e:
            print("OpenAI classify error:", e)
            classified = []
    return {
        "ticker": row['Symbol'],
        "halt_time": row['halt_time'],
        "resume_time": row['resume_time'],
        "reason": row.get('reason',''),
        "bars": bars,
        "news": news_items,
        "news_classified": classified
    }

def enrich_halts_with_bars_and_news(halts_df, pre_minutes=60, post_minutes=60, bar_timespan="minute"):
    rows = []
    ensure_dir(cfg.get('output_dir','outputs'))
    i = 0
    for _,row in halts_df.iterrows():
        e = enrich_single_halt(row, pre_minutes=pre_minutes, post_minutes=post_minutes, timespan=bar_timespan)
        rows.append(e)
        i += 1
        # print(e)
        time.sleep(0.25)
    return pd.DataFrame(rows)
