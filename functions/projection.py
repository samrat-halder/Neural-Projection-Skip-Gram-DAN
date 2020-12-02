import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.random_projection import SparseRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
import scipy.sparse as sp

class WordTokenizer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        out = [
            [
                word
                for word in sentence.split(" ")
                if not len(word) == 0
            ]
            for sentence in X
        ]
        return out

class CountVectorizer3D(CountVectorizer):

    def fit(self, X, y=None):
        X_flattened_2D = sum(X.copy(), [])
        super(CountVectorizer3D, self).fit_transform(X_flattened_2D, y)  # can't simply call "fit"
        return self

    def transform(self, X):
        return [
            super(CountVectorizer3D, self).transform(x_2D)
            for x_2D in X
        ]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class FeatureUnion3D(FeatureUnion):
    
    def fit(self, X, y=None):
        X_flattened_2D = sp.vstack(X, format='csr')
        super(FeatureUnion3D, self).fit(X_flattened_2D, y)
        return self
    
    def transform(self, X): 
        return [
            super(FeatureUnion3D, self).transform(x_2D)
            for x_2D in X
        ]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

