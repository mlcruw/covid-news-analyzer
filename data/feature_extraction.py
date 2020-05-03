import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer


class FeatureExtractor:
    """
    Description
    ===========
    Tranforms text to features
    """
    
    def __init__(self, X, feat='none'):
        feat_mapping_dict = {
            'BoW': self.BoW,
            'Word2Vec': self.Word2Vec
        }
        self.out = feat_mapping_dict[feat](X)
        
    def BoW(self, X):
        """
        [sentence_1, sentence_2, ..., sentence_n] => dictionary
        """
        vectorizer = CountVectorizer()
        vectorizer.fit(X)
        return vectorizer.vocabulary_
    
    def Word2Vec(self, X):
        """
        nested list of words => nested list of "word embedding vector"
        """
        model = Word2Vec(sentences = X, size = 100, sg = 1, window = 3, min_count = 1, iter = 10)
        #https://github.com/buomsoo-kim/Word-embedding-with-Python/blob/master/word2vec/source%20code/word2vec.ipynb
        embed = [[[model[X[i][j]]] for j in range(len(X[i]))] for i in range(len(X))]
        return embed
