import openai
from .config import load
cfg = load()
openai.api_key = cfg.get('openai_api_key')

DEFAULT_CATEGORIES = ["NO_NEWS", "EARNINGS", "FDA", "IPO_SPAC", "ANALYST", "MERGER_OFFERING", "OTHER"]

def _build_prompt(article_title, article_body):
    prompt = f"""
        You are given a short news article headline and summary about a stock. Classify the news into one of these categories: {DEFAULT_CATEGORIES}.
        Return a JSON object with:
        - category: one of {DEFAULT_CATEGORIES}
        - reason: 1-2 sentence explanation why
        - sentiment: a number from -1 (very negative) to 1 (very positive), where negative means likely bearish for the stock.
        Here is the article:
        Title: {article_title}
        Summary: {article_body}
        JSON:
    """
    return prompt

def classify_single_article(article):
    """
    article is polygon news result (dict). Uses 'title' and 'description' or 'summary'.
    Returns dict with category, reason, sentiment and original headline/url/timestamp.
    """
    title = article.get('title','')
    desc = article.get('description') or article.get('summary') or article.get('article_url') or ''
    prompt = _build_prompt(title, desc)
    try:
        resp = openai.chat.completions.create(
            model=cfg['news'].get('openai_model','gpt-4o-mini'),
            messages=[{"role":"user","content":prompt}],
            temperature=cfg['news'].get('classify_temperature',0.0),
            max_tokens=cfg['news'].get('max_tokens',256)
        )
        text = resp.choices[0].message['content']
        import json, re
        m = re.search(r'\{.*\}', text, flags=re.S)
        if m:
            parsed = json.loads(m.group(0))
            return {
                "title": title,
                "published_utc": article.get('published_utc'),
                "url": article.get('article_url'),
                "category": parsed.get('category'),
                "reason": parsed.get('reason'),
                "sentiment": parsed.get('sentiment')
            }
        else:
            return _keyword_fallback(article)
    except Exception as e:
        return _keyword_fallback(article)

def _keyword_fallback(article):
    text = (article.get('title','') + ' ' + (article.get('description') or '')).lower()
    keys = cfg.get('news',{}).get('keywords',{})
    for k,arr in keys.items():
        for kw in arr:
            if kw in text:
                cat_map = {
                    'earnings':'EARNINGS',
                    'fda':'FDA',
                    'ipo':'IPO_SPAC',
                    'merger':'MERGER_OFFERING',
                    'offering':'MERGER_OFFERING'
                }
                return {"title": article.get('title'), "published_utc": article.get('published_utc'), "url": article.get('article_url'), "category": cat_map.get(k,'OTHER'), "reason":"keyword match", "sentiment":0.0}
    return {"title": article.get('title'), "published_utc": article.get('published_utc'), "url": article.get('article_url'), "category":"OTHER", "reason":"no match", "sentiment":0.0}

def classify_news_items(news_items):
    classified = []
    for n in news_items:
        try:
            c = classify_single_article(n)
            classified.append(c)
        except Exception as e:
            classified.append({"title": n.get('title'), "category": "OTHER", "reason": "error", "sentiment": 0.0})
    return classified
