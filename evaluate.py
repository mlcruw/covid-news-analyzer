import pandas as pd
from nltk import sent_tokenize
from scipy import stats
import pickle

def eval_emotion(article_text):
    with open('test.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)
    sentences = sent_tokenize(article_text)
    pred_emotions = loaded_pipeline.predict(sentences)
    final_emotion = stats.mode(pred_emotions).mode[0]
    return final_emotion

if __name__=="__main__":
    data = pd.read_csv('covid19_articles/covid_19_articles.csv')
    article0 = data['text'][0]
    print(eval_emotion(article0))
