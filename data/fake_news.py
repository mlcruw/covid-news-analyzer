import glob
import zipfile
import pandas
from .dataset import Dataset


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
    self.dataset_dir = super().download_data(self.dirname, self.cmd)


  # Read data from files
  def read_data(self):
    print("Reading data...")
    self.train_data = pandas.read_csv(self.dataset_dir + "/train.csv", sep=",")
    self.test_data = pandas.read_csv(self.dataset_dir + "/test.csv", sep=",")
    print("Done")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")


  # Print string for class object
  def __str__(self):
    return self.__class__.name
