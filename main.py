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

# Split the data
dataset.split_data(
  train_size=args.split_ratio,
  test_size=args.test_ratio)

config = Config(dataset=args.dataset,
                model=args.models,
                feats=args.feats,
                save_path=args.save_path,
                continue_train=args.continue_train,
                load_path=args.load_path,
                test=args.test_only)

trainer = Trainer(dataset=dataset, models=args.models, transforms=args.feats, cfg=config)
# If we only want to test
if args.test_only:
    trainer.load_model(args.models[0], args.feats[0], args.load_path)

# Train
if not(args.test_only):
    trainer.train()
# Get the evaluation metric
metrics = trainer.evaluate()
# Save best
if not(args.test_only):
    trainer.save_best(metrics)
    print("Simultaneously training done")
else:
    print("Test result : ")
    print(metrics)
    trainer.logger.info(metrics)
s
# Sample Usage:
# [Emotion]
# 1. Train
#       python3 main.py --dataset emo_aff --models lr linearsvm gnb --feats bow ngram tfidf --split_ratio 0.8 --test_ratio 0.1 --save_path emo.model
# 2. Test
#       python3 main.py --dataset emo_aff --models linearsvm  --feats ngram  --split_ratio 0.7 --test_ratio 0.3 --load_path emo.model -test_only

# [News Category]
# 1. Train
#       python3 main.py --dataset news_cat --models lr linearsvm gnb --feats bow ngram tfidf --split_ratio 0.02 --test_ratio 0.05 --save_path news.model
# 2. Test
#       python3 main.py --dataset news_cat --models lr --feats ngram --split_ratio 0.1 --test_ratio 0.2 --load_path news.model -test_only

