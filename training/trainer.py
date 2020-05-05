import os
# import os.path as os.path
from copy import copy
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
# from config import cfg
from data.feature_extraction import FeatureExtractor
import pickle

model_class_map = {'gnb': GaussianNB, 'mnb': MultinomialNB,\
    'svm': SVC, 'lr': LogisticRegression, 'linearsvm': LinearSVC}


class Trainer:
    def __init__(self, data=None, models=None, feat='none'):
        self.models = {}
        if data is not None:
            self.set_data(data)
        if models is not None:
            self.init_models(models)
        if feat != 'none':
            self.feat = feat
            self.preprocess_data()

    def set_data(self, data):
        """
        Setup data arrays that will be used for training and validation

        Args:
        data: dictionary of numpy arrays containing training and validation data
        """
        keys = ['X_train', 'y_train', 'X_val', 'y_val']
        for key in keys:
            if key not in data:
                raise Exception('Invalid data object passed to Trainer')
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
    
    def preprocess_data(self):
        self.X_train, self.X_val = FeatureExtractor(self.X_train, self.X_val, self.feat).out
        

    def init_models(self, models):
        """
        Setup model objects that will be trained

        Args:
        models: list of models that will be trained on the data
        """
        for model in models:
            self.add_model(model)
    
    def add_model(self, model):
        """
        Add (model, model_obj) to the models dictionary
        """
        if model not in self.models:
            self.models[model] = model_class_map[model]()
    
    def train(self):
        """
        Train all the models on training data
        """
        for model_path, model_obj in self.models.items():
            print("Training {}".format(model_path))
            model_obj.fit(self.X_train, self.y_train)
    
    def train_model(self, model):
        """
        Train a specific model on training data
        """
        if model not in self.models:
            raise Exception("{} doesn't exist in models".format(model))
        self.models[model].fit(self.X_train, self.y_train)
    
    
    def save_model(self, model, model_path):
        """
        Save a specific model to the path : "model_path"
        """
        if model not in self.models:
            raise Exception("{} doesn't exist in models".format(model))
            
        file_path = os.path.join(cfg.model_dir,model_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self.models[model], file)
    
    def load_model(self, model, model_path):
        """
        Load a specific model from the path : "model_path"
        """
        if model not in self.models:
            raise Exception("{} doesn't exist in models".format(model))
            
        file_path = os.path.join(cfg.model_dir,model_path)
        with open(file_path, 'rb') as file:
            self.models[model] = pickle.load(file)
    
    def evaluate(self):
        """
        Evaluate all the models on validation data and return metrics
        stored in a pandas DataFrame
        """
        metrics = []
        for model_path, model_obj in self.models.items():
            y_pred = model_obj.predict(self.X_val)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_val, y_pred, average='micro')
            metrics.append({'precision': precision, 'recall': recall, 'f1-score': f1})
        return pd.DataFrame(metrics, index=self.models.keys())

if __name__=='__main__':
    """
    Unit test
    """
    n_train = 100
    n_val = 20
    n_feats = 20

    X_train = np.random.rand(n_train, n_feats)
    y_train = np.random.rand(n_train)
    y_train[y_train>=0.5] = 1
    y_train[y_train<0.5] = 0


    X_val = np.random.rand(n_val, n_feats)
    y_val = np.random.rand(n_val)
    y_val[y_val>=0.5] = 1
    y_val[y_val<0.5] = 0

    """
    Test BoW
    """
    train_word_arr = ['the apple was good', 'I flew too close to the sun.']
    train_label = np.random.rand(2)
    data_test = {'X_train': train_word_arr, 'y_train': train_label, 'X_val': train_word_arr, 'y_val': train_label}
    trainer_test = Trainer(data=data_test, models=['gnb', 'svm'], feat='BoW')
    print('Test BoW done')

    """
    Test Word2Vec (gensim)
    """
    sentences = []
    sentences.append(['[', 'The', 'Tragedie', 'of', 'Hamlet', 'by', 'William', 'Shakespeare', '1599', ']'])
    sentences.append(['Actus', 'Primus', '.'])
    sentences.append(['Fran', '.'])
    train_label = np.random.rand(3)
    data_test = {'X_train': sentences, 'y_train': train_label, 'X_val': sentences, 'y_val': train_label}
    trainer_test = Trainer(data=data_test, models=['gnb', 'svm'], feat='Word2Vec')
    print('Test Word2Vec done')
    
   


    data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}

    trainer = Trainer(data=data, models=['gnb', 'svm'])
    # print(trainer.y_val)
    # print(trainer.models)

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    print("Adding lr")

    trainer.add_model('lr')
    trainer.train_model('lr')
    metrics = trainer.evaluate()
    print(metrics)