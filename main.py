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
from data.emotion_affect import EmotionAffectDataset

dataset_map = {
  'stan_sent': StanfordSentimentDataset,
  'news_cat': NewsCategoryDataset,
  'fake_news': FakeNewsDataset,
  'emo_aff': EmotionAffectDataset
}

# Parse arguments
args = argparser()
# print("Command line arguments:")
# print(args)

# Create the dataset class object
dataset = dataset_map[args.dataset]()
dataset.split_data(train_split = args.split_ratio, test_ratio = args.test_ratio)
data_inp = {'X_train': dataset.train_data['X'], 'y_train': dataset.train_data['y'], 'X_val': dataset.test_data['X'], 'y_val': dataset.test_data['y']}
# The above step takes care of reading the dataset
# and splitting it

# The declaration for the best precision and the best configuration
best_precision = 0.0
best_config = Config()

for model in args.models:
  # TODO: include for feat in args.feats here later

  # Idea is to have a one-to-one mapping between
  # Trainer object and Config object.
  # When saving model in Trainer obj, we could save
  # its corresponding Config object too

  #Note below the usage of [feat] and [model]
  #Added feature selection 
  for feat in args.feats:
      config = Config(dataset=args.dataset,
                  model=model,
                  feats=[feat],
                  save_path=args.save_path,
                  continue_train=args.continue_train,
                  load_path=args.load_path,
                  test=args.test_only)
      # print("Configuration:")
      # print(', '.join("%s: %s" % item for item in vars(config).items()))

      # TODO:
      # - writing everything as pandas dataframe - any downsides?
      # - trainer class
      #   - also create a one-to-one map between Train and Config objects
      # - metrics
      # - save model
      # - modify preprocessor function to use pd.series

      # TODO: passing the config object to trainer?
      # If not passed here, do not comment "import config" in trainer otherwise it does not know the folder path for loading and saving models

      # When calling like this, the mapping from Train and Config is already one-to-one?
      trainer = Trainer(data=data_inp, models=[model], feat=feat, cfg=config)
      #1. If continue train from a saved model
      #... example usage of continue_train
      #........python3 main.py --dataset news_cat --models lr --feats bow --split_ratio 0.02 --test_ratio 0.01 --load_path bow_0 --save_path bow_1 -c
      #... example usage of test_only (test PHASE)
      #........python3 main.py --dataset news_cat --models lr --feats bow --split_ratio 0.003 --test_ratio 0.01 --load_path bow_0 -test_only
      if args.continue_train or args.test_only:
         #TO-DO: it can log sth. like "continue training from" or "testing on"
         trainer.load_model(model, args.load_path)
      
      #2. train
      if not(args.test_only):
         trainer.train()

      #3. Metrics evaluation
      metrics = trainer.evaluate()
      print(metrics)

      #4. Update the best acc and config
      if metrics.iloc[0]['precision'] > best_precision:
         best_precision = metrics.iloc[0]['precision']
         best_config = config

      #5. Of course output the evaluation result
      trainer.logger.info('The evaluation result is')
      trainer.logger.info(metrics.to_string())

      #6. save model
      if not(args.test_only):
         trainer.save_model(model, args.save_path)

#Output the best model precision and configuration
trainer.logger.info(['The best model precision is %.2f ' % best_precision])
trainer.logger.info(['The best model configuration is ', ', '.join("%s: %s" % item for item in vars(best_config).items())])

#Sample usage:
#python3 main.py --dataset news_cat --models lr linearsvm --feats bow word2vec  --split_ratio 0.005 --test_ratio 0.0005 --save_path best