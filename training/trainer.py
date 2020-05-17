import os
# import os.path as os.path
from copy import copy
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, precision_recall_fscore_support
#Do not comment as it contains the default configuration for trainer, see parameter for __init__ below
from training.config import Config
from data.feature_extraction import FeatureExtractor
import pickle
#The logging system
from utils.logger import colorlogger


model_class_map = {
    'mnb': MultinomialNB, 'svm': SVC, 'mlp': MLPClassifier,
    'lr': LogisticRegression, 'linearsvm': LinearSVC, 'xgb': XGBClassifier,
    'rf': RandomForestClassifier, 'ada': AdaBoostClassifier
}

model_params = {
    'svm': {'C': [1e-1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']},
    'lr': {'C': [1e-2, 1e-1, 1, 10], 'max_iter': [1000]},
    'rf': {'max_depth': [None, 10, 20, 50], 'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]},
    'ada': {'n_estimators': [20, 50, 100]},
    'xgb': {'eta': [0.1, 0.3], 'max_depth': [4, 6],
        'lambda': [1e-2, 1e-1, 1]}
}

transform_class_map = {
    'bow': CountVectorizer,
    'tfidf': TfidfVectorizer,
    'ngram': CountVectorizer
}

transform_params = {
    'bow': {'stop_words': 'english', 'min_df': 5, 'max_features': 5000},
    'tfidf': {'stop_words': 'english', 'min_df': 5, 'max_features': 5000},
    'ngram': {
        'stop_words': 'english', 'min_df': 5, 'max_features': 5000,
        'ngram_range': (1, 3)
    }
}

# Important: TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


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
    #TODO: the config and log_name as arguments?
    def __init__(self, dataset, models, transforms, cfg=Config(), log_name='logs.txt', grid=True):
        self.pipelines = {}
        self.gridsearch = {}
        self.models = models
        self.transforms = transforms
        self.dataset = dataset
        self.init_pipelines(models, transforms)
        self.grid = grid
        if self.grid:
            self.init_gridsearch()
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
            self.transformed[transform]['X_test'] = X_val

    def get_train_data(self):
        """
        Returns X_train, y_train
        """
        return self.dataset.train_data['X'].values, self.dataset.train_data['y'].values

    def get_test_data(self):
        """
        Returns X_test, y_test
        """
        return self.dataset.test_data['X'].values, self.dataset.test_data['y'].values

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
        #print('transform params: ', transform_params)
        transform_obj = transform_class_map[transform](**tranform_params)
        if model == 'lr':
            model_obj = model_class_map[model](max_iter=1000)
        elif model == 'svm':
            model_obj = OneVsRestClassifier(model_class_map[model]())
        elif model in ['rf', 'xgb']:
            model_obj = model_class_map[model](n_jobs=-1)
        else:
            model_obj = model_class_map[model]()
        # Fix the "to dense" bug "fit" requires dense input
        steps = [('tranform', transform_obj),
                #  ('to_dense', DenseTransformer()),
                #  ('pca', PCA(n_components=1000)),
                 ('model', model_obj)]
        self.pipelines[model][transform] = Pipeline(steps, verbose=True)

    def get_param_grid(self, model, transform):
        if model not in model_params:
            # Use default params in param_grid for consistency with other models
            model_obj = self.pipelines[model][transform].named_steps['model']
            param_grid = {'model__' + k: [v] for k, v in model_obj.get_params().items()}
        else:
            param_grid = {}
            for param in model_params[model]:
                pname = 'model__{}'.format(param)
                pval = model_params[model][param]
                param_grid[pname] = pval
        return param_grid

    def init_gridsearch(self, n_jobs=-1):
        """
        Setup GridSearchCV over the pipeline objects for hyperparamter tuning
        """
        for model in self.pipelines:
            self.gridsearch[model] = {}
            for transform in self.pipelines[model]:
                param_grid = self.get_param_grid(model, transform)
                search = GridSearchCV(self.pipelines[model][transform],
                    param_grid, n_jobs=n_jobs, verbose=2)
                self.gridsearch[model][transform] = search

    def train(self):
        """
        Train all the models on training data with optional gridsearch

        Args:
            - grid (bool), default=True: If True, runs GridSearch over hyperparams
        """
        if self.grid:
            model_dict = self.gridsearch
        else:
            model_dict = self.pipelines

        for model in model_dict:
            for transform in model_dict[model]:
                model_obj = model_dict[model][transform]
                X_train, y_train = self.get_train_data()
                print("Training {} with {} transformation".format(model, transform))
                self.logger.info("Training {} with {} transformation".format(model, transform))
                model_obj.fit(X_train, y_train)

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
        Save a specific model and transformation to the path : "model_path"

        Args:
            - grid (bool), default=True: If True, saves the best_estimator from GridSearch
        """
        if model not in self.pipelines:
            raise Exception("{} doesn't exist in pipelines".format(model))
        if transform not in self.pipelines[model]:
            raise Exception("{} transformation doesn't exist in pipelines".format(transform))
        #It uses the configuration to find the model directory
        #Config should be imported somehow so that it knows where the folder is
        file_path = os.path.join(self.cfg.model_dir,model_path)
        with open(file_path, 'wb') as fw:
            if self.grid:
                model_obj = self.gridsearch[model][transform].best_estimator_
            else:
                model_obj = self.pipelines[model][transform]
            pickle.dump(model_obj, fw)

        # Output the associated config object to log
        self.logger.info("Saving configuration:")
        self.logger.info(', '.join("%s: %s" % item for item in vars(self.cfg).items()))

    def load_model(self, model, transform, model_path):
        """
        Load a specific model with transformation from the path : "model_path"
        """
        if model not in self.pipelines:
            raise Exception("{} doesn't exist in models".format(model))
        if transform not in self.pipelines[model]:
            raise Exception("{} transformation doesn't exist in pipelines".format(transform))
        #It uses the configuration to find the model directory
        #Config should be somewhere called to see the folder path
        file_path = os.path.join(self.cfg.model_dir,model_path)
        with open(file_path, 'rb') as file:
            self.pipelines[model][transform] = pickle.load(file)

        # Tell what the loading configuration looks like
        self.logger.info("Loading from this configuration:")
        self.logger.info(', '.join("%s: %s" % item for item in vars(self.cfg).items()))

    def evaluate(self):
        """
        Evaluate all the models on validation data and return metrics
        stored in a pandas DataFrame

        Args:
            - grid (bool), default=True: If True, uses the best_estimator from GridSearch
        """
        if self.grid:
            model_dict = self.gridsearch
        else:
            model_dict = self.pipelines

        metrics = []
        for model in model_dict:
            for transform in model_dict[model]:
                model_obj = model_dict[model][transform]
                X_test, y_test = self.get_test_data()
                y_pred = model_obj.predict(X_test)
                precision_micro = precision_score(y_test, y_pred, average='micro')
                precision_macro, recall_macro, f1_macro, _ = \
                    precision_recall_fscore_support(y_test, y_pred, average='macro')
                acc = accuracy_score(y_test, y_pred)
                metric_dict = {
                    'model': model,
                    'transform': transform,
                    'precision': precision_micro,
                    'precision (macro)': precision_macro,
                    'recall (macro)': recall_macro,
                    'f1-score (macro)': f1_macro,
                    'accuracy': acc
                }
                metrics.append(metric_dict)
        return pd.DataFrame(metrics)

    def save_best(self, metrics, save_path=None):
        """
        Save the best model to the path : "model_path"

        Args:
            - save_path (str), default=cfg.save_path: The save path for the best model
            - grid (bool), default=True: If True, saves the best_estimator from GridSearch
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
        if self.grid:
            best_config.params = self.gridsearch[best_config.model][best_config.feats].best_params_
            self.logger.info(['The best parameters found via GridSearch are {}'.format(best_config.params)])
        self.logger.info(['The best model configuration is ', ', '.join("%s: %s" % item for item in vars(best_config).items())])
        self.logger.info(metrics)
        # Save model to file
        if save_path is None:
            save_path = self.cfg.save_path
        self.save_model(metrics.iloc[best_row]['model'], metrics.iloc[best_row]['transform'], save_path)
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
