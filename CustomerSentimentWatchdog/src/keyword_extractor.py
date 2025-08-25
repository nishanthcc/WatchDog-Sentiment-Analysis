import yake
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordExtractor:
    def __init__(self):
        self.kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)

    def extract_yake(self, texts):
        joined = "\n".join([t for t in texts if isinstance(t, str)])
        kws = self.kw_extractor.extract_keywords(joined)
        return [k for k,_ in sorted(kws, key=lambda x: -x[1])][:10]

    def extract_tfidf(self, texts, topn=10):
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return []
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
        X = vec.fit_transform(texts)
        means = X.mean(axis=0).A1
        vocab = vec.get_feature_names_out()
        inds = means.argsort()[::-1][:topn]
        return [vocab[i] for i in inds]
