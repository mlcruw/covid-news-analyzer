import glob
import zipfile
import numpy as np
import pandas as pd
from .dataset import Dataset

# FIXME:
# - test_data.y has NaNs filled in it. Since dataset was downloaded from kaggle
#   don't know test_data labels


class FakeNewsDataset(Dataset):
  name = "Fake News Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    self.cmd = 'kaggle competitions download -c fake-news'
    self.dirname = 'fake_news_dataset'

    # Download the dataset to dirname using cmd
    self.download_data()

    # Read data from dirname
    # Note: this is different for different datasets
    # So make sure to change for each dataset
    self.read_data()

    # self.standardize_data()


  # Download the dataset
  def download_data(self):
    # Download and store the absolute path to dirname
    super().download_data(self.dirname, self.cmd)


  # Read data from files
  def read_data(self):
    print("\nReading data...")
    train_raw_data = pd.read_csv(self.dataset_dir + "/train.csv", sep=",")
    test_raw_data = pd.read_csv(self.dataset_dir + "/test.csv", sep=",")

    self.raw_data = pd.concat([train_raw_data, test_raw_data])

    self.train_data = pd.DataFrame()
    self.train_data['X'] = train_raw_data.title + train_raw_data.text
    self.train_data['y'] = train_raw_data.label

    self.test_data = pd.DataFrame()
    self.test_data['X'] = test_raw_data.title + test_raw_data.text
    self.test_data['y'] = pd.Series(np.nan, index=np.arange(len(test_raw_data.index)))

    # self.data = pd.concat([self.train_data, self.test_data])

    print("Done")


  # Split data
  def split_data(self, **kwargs):
    # Don't split data again, it's already split
    pass


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")
