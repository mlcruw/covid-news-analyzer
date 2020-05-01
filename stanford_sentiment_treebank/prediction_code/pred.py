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


#(0.42, 0.5, array([3, 4, 5])
#4.2
data_sample =  "What all of the sources agree about is the extensive cover-up of data and information about COVID-19 orchestrated by the Chinese government."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.59, 0.5, array([7, 6, 5])
#5.9
data_sample =  "extensive cover-up orchestrated by the Chinese government."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.6, 0.6, array([7, 5, 6])
#6
data_sample = "Asked by Fox News' John Roberts about the reporting, President Trump remarked at Wednesday's coronavirus press briefing,  More and more we're hearing the story...we are doing a very thorough examination of this horrible situation. "
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.4, 0.4, array([3, 5, 4])
#4
data_sample = "thorough examination of this horrible situation"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.53, 0.5, array([4, 6, 5])
#5.3
data_sample = "thorough examination of this"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.29, 0.3, array([4, 2, 3])
#2.9
data_sample = " This is tremendous,  said Zuo-Feng Zhang, an epidemiologist at the University of California, Los Angeles.  If they took action six days earlier, there would have been much fewer patients and medical facilities would have been sufficient. We might have avoided the collapse of Wuhan’s medical system. "
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.41, 0.4, array([3, 5, 4])
#4.1
data_sample = "A protest of more than 200 demonstrators broke out in Southern California on Friday against the state’s stay-at-home-orders in reaction to the coronavirus outbreak, according to reports."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.43, 0.5, array([3, 4, 5])
#4.3
data_sample = "A protest of more than 200 demonstrators broke out"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.2, 0.3, array([1, 2, 3]) ??? "We don't have danger?"
#2?
data_sample = " I don’t think there’s any reason for us to be on lockdown now,  Paula Doyle, 62, of Costa Mesa told the Los Angeles Times.  We didn’t have any dangers. We have no danger in our hospitals now of overflowing. "
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.42, 0.4, array([3, 5, 4])
#4.2
data_sample = " be on lockdown now"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.41, 0.4, array([3, 5, 4])
#4.1
data_sample = " not on lockdown now"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.45, 0.5, array([3, 4, 5])
#4.5
data_sample = "any dangers."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.43, 0.5, array([3, 4, 5])
#4.3
data_sample = "no danger"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.41, 0.4, array([3, 5, 4])
#4.1
data_sample = "danger"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.39, 0.4, array([5, 3, 4])
#3.9
data_sample = " We didn’t have any dangers. We have no danger"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.4, 0.4, array([3, 5, 4])
#4
data_sample = " don’t think there’s any reason for us to be on lockdown now"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.6, 0.7, array([6, 5, 7]) no help?
#6
data_sample = "They keep on saying that there are no free spots,  she said.  They didn’t provide me with any help. They’ve just made me wait. "
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.4, 0.4, array([3, 5, 4])
#4
data_sample = "They didn’t provide me with any help. They’ve just "
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.51, 0.5, array([4, 6, 5])
#5.1
data_sample = "They provide me with lots of help."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.42, 0.5, array([3, 4, 5])
#4.2
data_sample = "no free spots"
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.21, 0.3, array([1, 2, 3])
#2.1
data_sample = "Desperate to find help, Ms. Zhang and her family called everyone they could think of. But the hospitals were all full. Emergency responders told them they needed to secure a hospital bed first before an ambulance could be sent."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.3, 0.3, array([4, 2, 3])
#3
data_sample = "But they had no time to mourn. Ms. Zhang’s grandmother was now deteriorating rapidly. They took her to a hospital, where a doctor said that her lungs appeared on a CT scan as almost entirely white — signs of severe pneumonia. She later tested positive for the coronavirus."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.41, 0.5, array([4, 3, 5]) uncertain
#4.1
data_sample = "The damage is still highly uncertain. But if large gatherings like conferences and concerts continue to be canceled, and more people decide they will not fly this summer and stay home more generally, it’s likely to cripple the consumer-driven side of the economy."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.58, 0.6, array([7, 5, 6]) praise
#5.8
data_sample = "Trump went on to praise Democratic governors who have complimented parts of the federal response, including Cuomo, Louisiana Gov. John Bel Edwards and California Gov. Gavin Newsom. He then played the clip of Cuomo describing how the President sent in the Army Corps of Engineers to build 2,500 beds at the Javits Center in New York."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.4, 0.4, array([5, 3, 4]) struggling
#4.0
data_sample = "China is struggling to maintain its position in global value chains. Rising costs and increasingly unfriendly business environment are causing many potential investors to drag their feet."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.5, 0.5, array([6, 4, 5])
#5
data_sample = "While recent supportive policies for manufacturing, small businesses and industries heavily affected by the epidemic have had a more obvious effect on the manufacturing sector, it is more difficult for service companies to make up their cash flow losses."
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

#(0.31, 0.3, array([2, 4, 3])
#3.1
data_sample = "it is more difficult for "
cur_result = live_test(prd_model,data_sample, word_idx)
print(cur_result)

def main():
    f = open('../../oldgit/covid_19_articles.sentences', 'r')
    f_res = open('result.txt', 'w')
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
            f_res.write("Data sample is ")
            f_res.write('\n')
            f_res.write(data_sample)
            f_res.write('\n')
            cur_result = live_test(prd_model,data_sample, word_idx)
            #top class is 1-based 1-10 10: most positive
            top_class = cur_result[2][2] + 1 
            top_class_cnt[top_class] += 1
            weighted_top_class += top_class

            single_score_dot_avg += cur_result[0]
            f_res.write(str(cur_result[0]))
            f_res.write('\n')
            f_res.write('\n')
            f_res.write('=======================================================================\n')
            f_res.write('\n')
            f_res.write('\n')
            
        weighted_top_class /= float(sentence_num)
        single_score_dot_avg /= float(sentence_num)

        print('Article ', cur_article)
        #print('     Average top class   ', int(weighted_top_class))
        print('     Weighted score      ', int(single_score_dot_avg * 10.0))

        most_frequent_class = 0
        for i in range(1, 11):
            if top_class_cnt[i] > top_class_cnt[most_frequent_class]:
                most_frequent_class = i
            print('            Class ', '%3d ' % i, ' percentage: ', top_class_cnt[i] / float(sentence_num))
        #print('     Most frequent class ', most_frequent_class)

        weighted_top_class = 0.0
        for i in range(1, 11):
            weighted_top_class += top_class_cnt[i] / float(sentence_num) * float(i)
        #print("     Weighted:           ", int(weighted_top_class))
    f.close()
    f_res.close()

main()
