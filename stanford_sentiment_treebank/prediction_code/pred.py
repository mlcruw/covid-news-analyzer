#from pycorenlp import StanfordCoreNLP
#import os
#os.chdir('/app-sentiment-algo')

import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
import h5py
#import utility_functions as uf
from keras.models import model_from_json
from keras.models import load_model
import json
from nltk.tokenize import RegexpTokenizer



def live_test(trained_model, data, word_idx):
    live_list = []
    live_list_np = np.zeros((56,1))
    # split the sentence into its words and remove any punctuations.
    tokenizer = RegexpTokenizer(r'\w+')
    data_sample_list = tokenizer.tokenize(data)
    if len(data_sample_list) > 56:
        data_sample_list = data_sample_list[:56]
    labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
    # get index for the live stage
    data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
    data_index_np = np.array(data_index)
    # padded with zeros of length 56 i.e maximum length
    padded_array = np.zeros(56)
    padded_array[:data_index_np.shape[0]] = data_index_np
    data_index_np_pad = padded_array.astype(int)
    live_list.append(data_index_np_pad)
    live_list_np = np.asarray(live_list)


    # get score from the model
    score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
    #print('score: ', score)

    single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band
    # weighted score of top 3 bands
    top_3_index = np.argsort(score)[0][-3:]
    top_3_scores = score[0][top_3_index]
    top_3_weights = top_3_scores/np.sum(top_3_scores)

    #print('top 3 index: ', top_3_index)
    #print('top 3 scores: ', top_3_scores)
    #print('top 3 weights: ', top_3_weights)

    single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)
    return single_score_dot, single_score, top_3_index, top_3_scores, top_3_weights

weight_path = 'model/best_model.hdf5'
prd_model = load_model(weight_path)
prd_model.summary()
word_idx = json.load(open("Data/word_idx.txt"))

# sample sentence
#data_sample = "And yet he spent much of the holiest weekend on the Christian calendar in an uproar over crushing news reports that make it clear his early response to coronavirus warnings was a failure — that cost thousands of human lives."

#data_sample = "“This does not mean output is now back to its pre-virus trend,” said Julian Evans-Pritchard, senior China economist at Capital Economics, in a note. “Instead, it simply suggests that economic activity improved modestly relative to February’s dismal showing, but remains well below pre-virus levels. This is consistent with what the daily activity indicators show.”"
#result = live_test(prd_model,data_sample, word_idx)

#print(result)




def main():
    f = open('../../oldgit/covid_19_articles.sentences', 'r')
    cur_article = 0
    while True:
        cur_article += 1
        sentence_num = f.readline()
        sentence_num = int(sentence_num)
        sentences = []
        for rows in range(sentence_num):
            sentence = f.readline()
            br = f.readline()
            #print(sentence)
            #print(br)
            sentences.append(sentence)
        #print(sentences)

        br = f.readline()
        #print('\n\n')
        if sentence_num == -1:
            break
        top_class_cnt = np.zeros(11)
        weighted_top_class = 0
        single_score_dot_avg = 0
        for i in range(sentence_num):
            data_sample = sentences[i]
            #print("        data sample is ")
            #print(data_sample)
            cur_result = live_test(prd_model,data_sample, word_idx)
            #top class is 1-based 1-10 10: most positive
            top_class = cur_result[2][2] + 1 
            top_class_cnt[top_class] += 1
            weighted_top_class += top_class

            single_score_dot_avg += cur_result[0]
            #print(cur_result)
            #print('        Top class is (1-based) : ', top_class)
        weighted_top_class /= float(sentence_num)
        single_score_dot_avg /= float(sentence_num)

        print('Article ', cur_article)
        print('     Average top class   ', int(weighted_top_class))
        print('     Single score dotavg ', int(single_score_dot_avg * 10.0))

        most_frequent_class = 0
        for i in range(1, 11):
            if top_class_cnt[i] > top_class_cnt[most_frequent_class]:
                most_frequent_class = i
            print('            Class ', '%3d ' % i, ' percentage: ', top_class_cnt[i] / float(sentence_num))
        print('     Most frequent class ', most_frequent_class)

        weighted_top_class = 0.0
        for i in range(1, 11):
            weighted_top_class += top_class_cnt[i] / float(sentence_num) * float(i)
        print("     Weighted:           ", int(weighted_top_class))
    f.close()

main()
