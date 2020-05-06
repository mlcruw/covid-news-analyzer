import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import numpy as np

class FeatureExtractor:
    """
    Description
    ===========
    Tranforms text to features
    """
    
    def __init__(self, X_train, X_val, feat='none'):
        feat_mapping_dict = {
            'bow': self.bow,
            'word2vec': self.word2vec,
            'tfidf': self.tfidf,
            'ngram': self.ngram
        }
        self.out = feat_mapping_dict[feat](X_train, X_val)
        
    def bow(self, X_train, X_val):
        """
        [sentence_1, sentence_2, ..., sentence_n] => dictionary
        """
        vectorizer = CountVectorizer()
        #vectorizer.fit(X)
        X_train_counts = vectorizer.fit_transform(X_train)
        #return vectorizer.vocabulary_
        X_val_counts = vectorizer.transform(X_val)
        return X_train_counts.toarray(), X_val_counts.toarray() #Fit requires dense
    
    def tfidf(self, X_train, X_val):
        """
        [sentence_1, sentence_2, ..., sentence_n] => dictionary
        """
        vectorizer = TfidfVectorizer()
        X_train_counts = vectorizer.fit_transform(X_train)
        X_val_counts = vectorizer.transform(X_val)
        return X_train_counts.toarray(), X_val_counts.toarray() #Fit requires dense
    
    def ngram(self, X_train, X_val):
        """
        [sentence_1, sentence_2, ..., sentence_n] => dictionary
        """
        #TODO 5 is large I guess?
        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
        X_train_counts = vectorizer.fit_transform(X_train)
        #print(vectorizer.get_feature_names())
        X_val_counts = vectorizer.transform(X_val)
        return X_train_counts.toarray(), X_val_counts.toarray() #Fit requires dense
        
    
    #TODO: load a pretrained embedding model
    def word2vec(self, X_train, X_val):
        """
        nested list of words => nested list of "word embedding vector"
        """
        embed_dim = 30
        max_word = 30
        model = Word2Vec(sentences = X_train, size = embed_dim, sg = 1, window = 3, min_count = 1, iter = 30)
        pretrained_weights = model.wv.syn0
        vocab_size, embedding_size = pretrained_weights.shape
        print('Vocab size ', vocab_size, ' embed shape ', embedding_size)
        
        #https://github.com/buomsoo-kim/Word-embedding-with-Python/blob/master/word2vec/source%20code/word2vec.ipynb
        embed_train = np.zeros((len(X_train), max_word, embed_dim))
        for i in range(len(X_train)):
            for j in range(max_word):
                for k in range(embed_dim):
                    if j < len(X_train[i]):
                        embed_train[i][j][k] = model[X_train[i][j]][k]
                        
                        
        embed_train = embed_train.reshape((len(X_train), max_word * embed_dim))
                
        #embed_train = [[[model[X_train[i][j]] if j < len(X_train[i]) else 0] for j in range(56)] for i in range(len(X_train))]
        
        #embed_train = np.array(embed_train)
        print('Embed train . shape' , embed_train.shape)
        
        #embed_val = [[[(model[X_val[i][j]] if X_val[i][j] in model.wv.vocab else 0) if j < len(X_val[i]) else 0] for j in range(56)] for i in range(len(X_val))]
        embed_val = np.zeros((len(X_val), max_word, embed_dim))
        for i in range(len(X_train), len(X_train) + len(X_val)):
            for j in range(max_word):
                for k in range(embed_dim):
                    if j < len(X_val[i]):
                        if X_val[i][j] in model.wv.vocab:
                            embed_val[i - len(X_train)][j][k] = model[X_val[i][j]][k]
        embed_val = embed_val.reshape((len(X_val), max_word * embed_dim))
        print('Embed val . shape' , embed_val.shape)
        
        return embed_train, embed_val
