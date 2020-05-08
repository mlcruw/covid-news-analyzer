import glob
import zipfile
import pandas as pd
import json
from .dataset import Dataset
from sklearn.preprocessing import LabelEncoder

# TODO:


class EmotionAffectDataset(Dataset):
  #TODO: change the name of this dataset
  name = "Emotion Affect Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    #TODO: write code to download the data; it's not on kaggle
    self.dirname = 'emotion_affect_dataset'

    # Download the dataset to dirname using cmd
    #self.download_data()

    # Read data from dirname
    # Note: this is different for different datasets
    # So make sure to change for each dataset
    self.read_data()

    # self.standardize_data()
    # self.split_data()


  # Download the dataset
  def download_data(self):
    super().download_data(self.dirname, self.cmd)


  # Read data from files
  def read_data(self):
    print("Reading data...")
    #TODO!!!!!! replace the hard-coded path
    #Some lines start with "the_tale_of" and has no "@"; just a caption; not a real training sample
    #use raw as extension because .txt will be filtered
    with open('data/datasets/emotion_affect_dataset/Emotion_Affect.raw', 'r') as f:
      # Read raw data into pandas dataframe
      s = f.readlines()
      emo_class = []
      emo_text = []
      emo_ind = []
      for line in s:
          if line.find('@') == -1:
            continue
          line = line.split('@') #the raw data has @ which is not , it is possible to transform to csv file but there are some lines starting with the_tale_of which is useless
          emo_ind.append(int(line[0])) #sentence_id
          emo_class.append(int(line[1]) - 1) #2-7 in raw data -> make it 1-6
          emo_text.append(line[2]) #the last char is '\n'
      proc_data = {'X': emo_text, 'y': emo_class, 'index': emo_ind}   
      
      # Create a new pandas dataframe from raw_data columns
      data = pd.DataFrame(proc_data)
        
      self.data = data
      print(len(self.data.index))

    print("Done")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("EmoAffe standardize_data method doesn't exist!")
