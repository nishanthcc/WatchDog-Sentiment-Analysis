import pandas as pd
import numpy as np
from datetime import datetime

def ensure_datetime(series, default_date=None):
    if series is None:
        return None
    def _parse(x):
        try:
            return pd.to_datetime(x)
        except Exception:
            return default_date or pd.Timestamp.today()
    return series.apply(_parse)

def compute_happiness_score(sentiment_scores):
    # Expect sentiment_scores in [-1,1] (VADER style) or [0,1] positive prob; normalize
    s = np.array(sentiment_scores, dtype=float)
    s = np.clip(s, -1, 1)
    s = (s + 1) / 2.0  # 0..1
    return (s.mean() * 100.0) if s.size else 0.0

def color_sentiment(val):
    if isinstance(val, str):
        v = val.lower()
        if v.startswith('neg'):
            return 'background-color: rgba(255, 0, 0, 0.15)'
        if v.startswith('pos'):
            return 'background-color: rgba(0, 255, 0, 0.15)'
    return ''
