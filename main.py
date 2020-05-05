import os
import pandas as pd
from training.argparser import argparser
from training.config import Config
from training.trainer import Trainer
import json
# from data.preprocessing import preprocessor_fn

from data.stanford_sentiment import StanfordSentimentDataset
from data.news_category import NewsCategoryDataset
from data.fake_news import FakeNewsDataset

dataset_map = {
  'stan_sent': StanfordSentimentDataset,
  'news_cat': NewsCategoryDataset,
  'fake_news': FakeNewsDataset
}

# Parse arguments
args = argparser()
# print("Command line arguments:")
# print(args)

# Create the dataset class object
dataset = dataset_map[args.dataset]()

# The above step takes care of reading the dataset
# and splitting it

for model in args.models:
  # TODO: include for feat in args.feats here later

  # Idea is to have a one-to-one mapping between
  # Trainer object and Config object.
  # When saving model in Trainer obj, we could save
  # its corresponding Config object too
  config = Config(dataset=args.dataset,
                  model=model,
                  feat=args.feat,
                  save_path=args.save_path,
                  continue_train=args.continue_train,
                  load_path=args.load_path,
                  test=args.test)
  # print("Configuration:")
  # print(', '.join("%s: %s" % item for item in vars(config).items()))

  # TODO:
  # - writing everything as pandas dataframe - any downsides?
  # - trainer class
  #   - also create a one-to-one map between Train and Config objects
  # - metrics
  # - save model
  # - modify preprocessor function to use pd.series


#   #==============================================================#
#   #1. BoW
#   text = list()
#   for i in range(df.shape[0]):
#     text.append(df.text[i])

#   train_text = text[0: train_num]
#   test_text = text[train_num: train_num + test_num]
#   train_label = [cate_dict[df.category[i]] for i in range(train_num)]
#   test_label = [cate_dict[df.category[i]] for i in range(train_num, train_num + test_num)]

#   data_inp = {'X_train': train_text, 'y_train': train_label, 'X_val': test_text, 'y_val': test_label}

#   trainer = Trainer(data=data_inp, models=cfg.model, feat='BoW')

#   #If continue train from a saved model
#   if cfg.continue_train or cfg.test_only:
#     trainer.load_model(cfg.model[0], 'bow_0')#cfg.load_path)
#   if not(cfg.test_only):
#     trainer.train()
#   metrics = trainer.evaluate()
#   print(metrics)

#   #save model
#   if not(cfg.test_only):
#       trainer.save_model(cfg.model[0], 'bow_1')#cfg.save_path)
#   #==============================================================#
#   #python3 main/train_news.py --dataset news_category_dataset --feat 'BoW' --model linearsvm

#   #python3 main/train_news.py --dataset news_category_dataset --feat 'BoW' --model lr --continue --load_path bow_0 --save_path bow_1


#   #==============================================================#
#   #2. Word2Vec (To-do: train this at a much larger scale Google News?)
#   #Tokenize
#   token = list()
#   for i in range(train_num + test_num):
#     token.append(preprocessor_fn(df.text[i], ['tokenize']))

#   train_token = token[0: train_num]
#   test_token = token[train_num: train_num + test_num]
#   train_label = [cate_dict[df.category[i]] for i in range(train_num)]
#   test_label = [cate_dict[df.category[i]] for i in range(train_num, train_num + test_num)]

#   data_inp = {'X_train': train_token, 'y_train': train_label, 'X_val': test_token, 'y_val': test_label}
#   trainer = Trainer(data=data_inp, models=cfg.model, feat=cfg.feat)

#   #If continue train from a saved model
#   if cfg.continue_train or cfg.test_only:
#       trainer.load_model(cfg.model[0], cfg.load_path)

#   if not(cfg.test_only):
#       trainer.train()
#   metrics = trainer.evaluate()
#   print(metrics)

#   # Save model
#   if not(cfg.test_only):
#     trainer.save_model(cfg.model[0], cfg.save_path)

#   #==============================================================#
#   #python3 main/train_news.py --dataset news_category_dataset --feat 'Word2Vec' --model lr --save_path word2vec_0

#   #python3 main/train_news.py --dataset news_category_dataset --feat 'Word2Vec' --model lr --load_path 'word2vec_0' --test
