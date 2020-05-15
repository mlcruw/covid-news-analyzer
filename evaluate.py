import pandas as pd
from nltk import sent_tokenize
#W/O these it does not work for me. If it works for you, kindly delete the two lines when executing.
import nltk
nltk.download('punkt')

from scipy import stats
import pickle

from sklearn import preprocessing
from data.emotion_affect import EmotionAffectDataset
from data.fake_news import FakeNewsDataset

# [IMPORTANT!!!!!!] Note some models require the entire article while others require the concatenation of headline title plus a separating space plus the body text. Make sure your modification aligns with those dataset classes.

def eval_emotion(article_text):
    # Evaluate emotion
    #   Input: The article
    #   Output: 'angry-disgusted' etc.
    with open('output/model_dump/emo.model', 'rb') as f:
        loaded_pipeline = pickle.load(f)
    sentences = sent_tokenize(article_text)
    pred_emotions = loaded_pipeline.predict(sentences)
    final_emotion = stats.mode(pred_emotions).mode[0]
    return EmotionAffectDataset.emotion_class_dict[final_emotion]

def eval_category(article_title_n_text):
    # Evaluate news category
    #   Input: article title + " " (space) + article body text
    #   Output: category e.g. "POLITICS"
    
    #load the label encoder 
    with open('data/datasets/news_category_dataset/label_encoder.pickle', 'rb') as f:
        news_encoder = pickle.load(f)
    
    with open('output/model_dump/news.model', 'rb') as f:
        loaded_pipeline = pickle.load(f)
    
    pred_cate = loaded_pipeline.predict(article_title_n_text)
    le = news_encoder
    return le.inverse_transform(pred_cate)[0]
    
def eval_fake(article_title_n_text):
    # Evaluate fakeness
    #   Input: same as eval_news title+" "+text
    #   Output: "Fake" or "Authentic"
    with open('output/model_dump/fake.model', 'rb') as f:
        loaded_pipeline = pickle.load(f)
        
    pred_fake = loaded_pipeline.predict(article_title_n_text)
    return FakeNewsDataset.fake_class_dict[pred_fake[0]]

def eval_sent(article_text):
    # Evaluate sentiment
    #   Input: same as eval_emotion: the entire article
    #   Output: sentiment score 0-4
    with open('output/model_dump/stan.model', 'rb') as f:
        loaded_pipeline = pickle.load(f)
    
    sentences = sent_tokenize(article_text)
    pred_sent = loaded_pipeline.predict(sentences)
    final_sent = stats.mode(pred_sent).mode[0]
    return final_sent

def evaluate_all(article_title, article_text):
    # Load all 4 models
    # Run predictions on all 4 models
    # Return a dictonary 
    # { "emotion": pred_emotion, "category": pred_category, "fake": pred_fake, "sentiment": pred_sentiment }
    # All pred_ are text strings except sentiment (a score \in [0, 4])
    
    # Special handling regarding title and text
    title_n_text = article_title + " " + article_text
    pred_emotion = eval_emotion(article_text)
    pred_category = eval_category([title_n_text]) #Don't delete []; required by sklearn
    pred_fake = eval_fake([title_n_text])
    pred_sentiment = eval_sent(article_text)
    return { "emotion": pred_emotion, "category": pred_category, "fake": pred_fake, "sentiment": pred_sentiment }
    
    
if __name__=="__main__":
    data = pd.read_csv('covid19_articles/covid_19_articles.csv')
    for i in range(0,20):
        print(' Article ', i)
        # Load all at once
        eval_cur = evaluate_all(data['title'][i], data['text'][i])
        print(eval_cur)
        
        print("--------------------------------------------\n")
        
        # 1. Emotion
        print(' 1. Emotion ')
        print(eval_cur["emotion"])
        
        # 2. News Category
        print(' 2. News Category')
        print(eval_cur["category"])
        
        # 3. Fake News
        print(' 3. Fake News')
        print(eval_cur["fake"])
        
        # 4. Sentiment
        print(' 4. Sentiment')
        print(eval_cur["sentiment"])
        print("============================================\n\n\n")
        