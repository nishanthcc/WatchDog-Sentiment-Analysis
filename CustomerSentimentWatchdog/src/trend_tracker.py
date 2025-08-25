import pandas as pd
from collections import Counter

class TrendTracker:
    def __init__(self):
        self.emotions = []

    def update(self, emotion: str):
        if emotion:
            self.emotions.append(emotion)

    def summary(self):
        return pd.Series(Counter(self.emotions))

    def check_alert(self, negative_set={'anger','sadness','fear','disgust','negative'}, threshold=0.4):
        if not self.emotions:
            return False, 0.0
        total = len(self.emotions)
        neg = sum(1 for e in self.emotions if e in negative_set or e.startswith('neg'))
        ratio = neg/total if total else 0.0
        return (ratio >= threshold), ratio
