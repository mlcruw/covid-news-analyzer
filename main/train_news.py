import os
import os.path as osp
import argparse
import pandas as pd
from trainer import Trainer
from config import cfg
import json
from data.preprocessing import preprocessor_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest='dataset')
    parser.add_argument('--model', nargs='+')
    parser.add_argument('--feat', type=str, dest='feat')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg.set_args(args.dataset, args.model, args.feat)
    
    print('Training on ', cfg.trainset)
    print('The path is ', osp.join(cfg.data_dir, cfg.trainset[0]))
    json_path = osp.join(cfg.data_dir, cfg.trainset[0])
    json_path = osp.join(json_path, 'News_Category_Dataset_v2.json')
    with open(json_path) as json_file:
        #Parse json category list
        df = pd.read_json(json_path, lines=True)
        print(df.head())
        cates = df.groupby('category')
        print("total categories:", cates.ngroups)
        print(cates.size())
        df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
        #use headline + description as input X
        df['text'] = df.headline + " " + df.short_description
        #print('The first sample: ', df.text[0])
        
        #Build the category -> class label dictionary for train/test label
        cate_dict = {}
        cate_id = 0
        for c in df.category:
            if not(c in cate_dict):
                cate_dict[c] = cate_id
                cate_id += 1
        
        #print(cate_dict)
        
        #TO-DO: These can be put into training/testing config in config.py
        train_num = 4000
        test_num = 1000
        
        #1. BoW
        text = list()
        for i in range(df.shape[0]):
            text.append(df.text[i])
    
        train_text = text[0: train_num]
        test_text = text[train_num: train_num + test_num]
        train_label = [cate_dict[df.category[i]] for i in range(train_num)]
        test_label = [cate_dict[df.category[i]] for i in range(train_num, train_num + test_num)]
        
        data_inp = {'X_train': train_text, 'y_train': train_label, 'X_val': test_text, 'y_val': test_label}
        
        trainer = Trainer(data=data_inp, models=cfg.model, feat='BoW')
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)
        
        #python3 main/train_news.py --dataset news_category_dataset --feat 'BoW' --model linearsvm
        
        #token = preprocessor_fn(df.text[0], tasks=['tokenize'])
        #print(token)
        
        #2. Word2Vec (To-do: train this at a much larger scale)
        #Tokenize
        token = list()
        for i in range(train_num + test_num):
            token.append(preprocessor_fn(df.text[i], ['tokenize']))
        
        train_token = token[0: train_num]
        test_token = token[train_num: train_num + test_num]
        train_label = [cate_dict[df.category[i]] for i in range(train_num)]
        test_label = [cate_dict[df.category[i]] for i in range(train_num, train_num + test_num)]
        
        data_inp = {'X_train': train_token, 'y_train': train_label, 'X_val': train_token, 'y_val': train_label}
        trainer = Trainer(data=data_inp, models=cfg.model, feat=cfg.feat)
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)
        
        #python3 main/train_news.py --dataset news_category_dataset --feat 'Word2Vec' --model lr
        

if __name__ == "__main__":
    main()
