import os
# import os.path as os.path
from copy import copy
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
#Do not comment as it contains the default configuration for trainer, see parameter for __init__ below
from training.config import Config
from data.feature_extraction import FeatureExtractor
import pickle
#The logging system
from utils.logger import colorlogger


model_class_map = {
    'gnb': GaussianNB, 'mnb': MultinomialNB,
    'svm': SVC, 'lr': LogisticRegression, 'linearsvm': LinearSVC
}

transform_class_map = {
    'bow': CountVectorizer,
    'tfidf': TfidfVectorizer,
    'ngram': CountVectorizer
}

transform_params = {
    'bow': {'stop_words':'english', 'min_df':5},
    'tfidf': {'stop_words':'english', 'min_df':5},
    'ngram': {
        'stop_words':'english', 'min_df':5,
        'analyzer':'char_wb', 'ngram_range':(5, 5)
    }
}

# Important: TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        return X.todense()

"""
#TODO:
- Add capability to save best model found during training
- Grid search for hyperparameters
"""
class Trainer:
    """
    Runs a cross product of feature transformations + models on the given
    dataset, evaluates all of them, and saves the best combination of
    transformation + model for the data.

    Usage:
        - trainer = Trainer(dataset, models, transforms)
        - traner.train()
        - results = trainer.evaluate()
    """
    #TODO: Pass dataset object here instead of data dict
    #TO-DO: the config and log_name as arguments?
    def __init__(self, dataset, models, transforms, cfg=Config(), log_name='logs.txt'):
        self.pipelines = {}
        self.models = models
        self.transforms = transforms
        self.dataset = dataset
        self.init_pipelines(models, transforms)
        # The current configuration for this trainer object
        self.cfg = cfg
        # The logger to saving (potentially) the config object and other stuff
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    def process_data(self, transforms):
        """
        Runs the feature transformations on the input data and create a dict
        containing features derived from all the transformations.
        """
        self.transformed = {}
        for transform in transforms:
            self.transformed[transform] = {}
            X_train, X_val = FeatureExtractor(self.dataset.train_data['X'],
                self.dataset.test_data['X'], transform).out
            self.transformed[transform]['X_train'] = X_train
            self.transformed[transform]['X_val'] = X_val

    def get_train_data(self):
        """
        Returns X_train, y_train
        """
        return self.dataset.train_data['X'].values, self.dataset.train_data['y'].values

    def get_test_data(self):
        """
        Returns X_test, y_test
        """
        return self.dataset.train_data['X'], self.dataset.train_data['y']

    def init_pipelines(self, models, transforms):
        """
        Setup pipeline objects that will be trained for each model x transform

        Args:
        models (list): list of models that will be trained on the data
        transforms (list): list of feature transformations to be applied on the data
        """
        for model in models:
            if model not in self.pipelines:
                self.pipelines[model] = {}
            for transform in transforms:
                self.add_pipeline(model, transform)

    def add_pipeline(self, model, transform):
        """
        Add a pipeline object with given transform and model
        """
        tranform_params = transform_params[transform]
        print('transform params: ', transform_params)
        transform_obj = transform_class_map[transform](**tranform_params)
        model_obj = model_class_map[model]()
        # Fix the "to dense" bug "fit" requires dense input
        steps = [('tranform', transform_obj),
                 ('to_dense', DenseTransformer()),
                 ('model', model_obj)]
        self.pipelines[model][transform] = Pipeline(steps, verbose=True)

    def train(self):
        """
        Train all the models on training data
        """
        for model in self.pipelines:
            for transform in self.pipelines[model]:
                pipeline_obj = self.pipelines[model][transform]
                X_train, y_train = self.get_train_data()
                print("Training {} with {} transformation".format(model, transform))
                self.logger.info("Training {} with {} transformation".format(model, transform))
                pipeline_obj.fit(X_train, y_train)

    def train_model(self, model):
        """
        Train a specific model on all transformations of training data
        """
        if model not in self.pipelines:
            raise Exception("{} doesn't exist in models".format(model))
        for transform in self.pipelines[model]:
            X_train, y_train = self.get_train_data()
            self.pipelines[model][transform].fit(X_train, y_train)

    def save_model(self, model, transform, model_path):
        """
        Save a specific model to the path : "model_path"
        """
        if model not in self.pipelines:
            raise Exception("{} doesn't exist in pipelines".format(model))
        if transform not in self.pipelines[model]:
            raise Exception("{} transformation doesn't exist in pipelines".format(transform))
        #It uses the configuration to find the model directory
        #Config should be imported somehow so that it knows where the folder is
        file_path = os.path.join(self.cfg.model_dir,model_path)
        with open(file_path, 'wb') as fw:
            pickle.dump(self.pipelines[model][transform], fw)

        # Output the associated config object to log
        self.logger.info("Saving configuration:")
        self.logger.info(', '.join("%s: %s" % item for item in vars(self.cfg).items()))

    # TODO: Won't work for multiple transforms - needs fix
    def load_model(self, model, model_path):
        """
        Load a specific model from the path : "model_path"
        """
        if model not in self.pipelines:
            raise Exception("{} doesn't exist in models".format(model))
        #It uses the configuration to find the model directory
        #Config should be somewhere called to see the folder path
        file_path = os.path.join(self.cfg.model_dir,model_path)
        with open(file_path, 'rb') as file:
            self.pipelines[model] = pickle.load(file)

        # Tell what the loading configuration looks like
        self.logger.info("Loading from this configuration:")
        self.logger.info(', '.join("%s: %s" % item for item in vars(self.cfg).items()))

    def evaluate(self):
        """
        Evaluate all the models on validation data and return metrics
        stored in a pandas DataFrame
        """
        metrics = []
        for model in self.pipelines:
            for transform in self.pipelines[model]:
                pipeline_obj = self.pipelines[model][transform]
                X_test, y_test = self.get_test_data()
                y_pred = pipeline_obj.predict(X_test)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='micro')
                metric_dict = {
                    'model': model,
                    'transform': transform,
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1
                }
                metrics.append(metric_dict)
        return pd.DataFrame(metrics)

    def save_best(self, metrics):
        """
        Save the best model to the path : "model_path"
        """
        best_precision = 0.0
        best_row = 0
        best_config = self.cfg
        for row in range(len(metrics)):
            if metrics.iloc[row]['precision'] > metrics.iloc[best_row]['precision']:
                best_row = row
        best_config.model = metrics.iloc[best_row]['model']
        best_config.feats = metrics.iloc[best_row]['transform']
        best_precision = metrics.iloc[best_row]['precision']
            
        # Output the best model precision and configuration
        self.logger.info(['The best model precision is %.2f ' % best_precision])
        self.logger.info(['The best model is %s ' % best_config.model])
        self.logger.info(['The best feature transformation is %s ' % best_config.feats])
        self.logger.info(['The best model configuration is ', ', '.join("%s: %s" % item for item in vars(best_config).items())])
        self.logger.info(metrics)
        # Save model to file
        self.save_model(metrics.iloc[best_row]['model'], metrics.iloc[best_row]['transform'], self.cfg.save_path)
        self.logger.info('Saving done')

if __name__=='__main__':
    """
    Unit test
    """
    pass
    # n_train = 100
    # n_val = 20
    # n_feats = 20

    # X_train = np.random.rand(n_train, n_feats)
    # y_train = np.random.rand(n_train)
    # y_train[y_train>=0.5] = 1
    # y_train[y_train<0.5] = 0


    # X_val = np.random.rand(n_val, n_feats)
    # y_val = np.random.rand(n_val)
    # y_val[y_val>=0.5] = 1
    # y_val[y_val<0.5] = 0

    # data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}

    # trainer = Trainer(data=data, models=['gnb', 'svm'], transforms=['bow'])
    # # print(trainer.y_val)
    # # print(trainer.models)

    # trainer.train()
    # metrics = trainer.evaluate()
    # print(metrics)

    # print("Adding lr")
    # trainer.add_model('lr', ['bow'])
    # trainer.train_model('lr')
    # metrics = trainer.evaluate()
    # print(metrics)
