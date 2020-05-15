import glob
import zipfile
import numpy as np
import pandas as pd
from .dataset import Dataset

# FIXME:
# - test_data.y has NaNs filled in it. Since dataset was downloaded from kaggle
#   don't know test_data labels


class StanfordSentimentDataset(Dataset):
  name = "Stanford Sentiment Analysis Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    self.cmd = 'kaggle competitions download -c sentiment-analysis-on-movie-reviews'
    self.dirname = 'stanford_sentiment'

    # Download the dataset to dirname using cmd
    self.download_data()

    # Read data from dirname
    # Note: this is different for different datasets
    # So make sure to change for each dataset
    self.read_data()

    # self.standardize_data()


  # Download the dataset
  def download_data(self):
    super().download_data(self.dirname, self.cmd)

    # Further extract zip files inside extracted directory
    downloaded_zipfiles = glob.glob(self.dataset_dir + '/*.zip')
    for ff in downloaded_zipfiles:
      with zipfile.ZipFile(ff, 'r') as z:
        z.extractall(self.dataset_dir)


  # Read data from files
  def read_data(self):
    print("\nReading data...")
    train_raw_data = pd.read_csv(self.dataset_dir + "/train.tsv", sep="\t")
    test_raw_data = pd.read_csv(self.dataset_dir + "/test.tsv", sep="\t")

    self.raw_data = pd.concat([train_raw_data, test_raw_data])

    data = pd.DataFrame()

    #self.train_data = pd.DataFrame()
    #self.train_data['X'] = train_raw_data.Phrase
    #self.train_data['y'] = train_raw_data.Sentiment

    #self.test_data = pd.DataFrame()
    #self.test_data['X'] = test_raw_data.Phrase
    #self.test_data['y'] = pd.Series(np.nan, index=np.arange(len(test_raw_data.index)))
    
    # Dataset is too large, use base class Dataset's train/split, o.w. the process will be killed by the terminal
    # Use "data" instead
    
    data['X'] = self.raw_data.Phrase
    data['y'] = self.raw_data.Sentiment
    self.data = data
    # self.data = pd.concat([self.train_data, self.test_data])

    print("Done")


  # Split data
  #def split_data(self, **kwargs):
  #  # Don't split data again, it's already split
  #  pass


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")
