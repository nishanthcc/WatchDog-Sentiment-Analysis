import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any
import numpy as np

# Fallbacks
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Transformers (optional, can be disabled)
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, mode='deep', multilingual=True):
        self.mode = mode  # 'deep' or 'fast'
        self.multilingual = multilingual
        self._init_pipelines()

    def _init_pipelines(self):
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_pipe = None
        self.emotion_pipe = None
        if self.mode == 'deep':
            try:
                # Sentiment model
                if self.multilingual:
                    # 1-5 star model => map to neg/neu/pos
                    self.sentiment_pipe = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
                else:
                    self.sentiment_pipe = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
                # Emotion model
                self.emotion_pipe = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=False)
            except Exception as e:
                # Fallback to fast mode silently
                self.mode = 'fast'

    def _vader_sentiment(self, text: str) -> Dict[str, Any]:
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        # crude emotion mapping
        emo = 'joy' if compound > 0.4 else ('anger' if compound < -0.4 else 'neutral')
        return {
            'sentiment': label,
            'sentiment_score': compound,
            'emotion': emo,
            'emotion_score': abs(compound)
        }

    def _map_multilingual_stars(self, label):
        # labels like '1 star', '2 stars'...
        num = int(label.split()[0]) if label and label[0].isdigit() else 3
        if num <= 2:
            return 'negative'
        elif num == 3:
            return 'neutral'
        else:
            return 'positive'

    def analyze_one(self, text: str) -> Dict[str, Any]:
        text = (text or '').strip()
        if not text:
            return {'sentiment': 'neutral', 'sentiment_score': 0.0, 'emotion': 'neutral', 'emotion_score': 0.0}

        if self.mode == 'deep' and self.sentiment_pipe is not None:
            try:
                s = self.sentiment_pipe(text)[0]
                if 'label' in s and 'star' in s['label']:
                    sentiment_label = self._map_multilingual_stars(s['label'])
                    sentiment_score = float(s.get('score', 0.5)) * (1.0 if sentiment_label == 'positive' else -1.0 if sentiment_label=='negative' else 0.0)
                else:
                    sentiment_label = s['label'].lower()
                    sentiment_score = s.get('score', 0.5) * (1.0 if sentiment_label=='positive' else -1.0 if sentiment_label=='negative' else 0.0)
                emo = self.emotion_pipe(text)[0] if self.emotion_pipe else {'label': 'neutral', 'score': 0.0}
                return {
                    'sentiment': sentiment_label,
                    'sentiment_score': float(sentiment_score),
                    'emotion': emo.get('label','neutral').lower(),
                    'emotion_score': float(emo.get('score',0.0))
                }
            except Exception:
                # fallback
                return self._vader_sentiment(text)
        else:
            return self._vader_sentiment(text)
