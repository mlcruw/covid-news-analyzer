import glob
import zipfile
import pandas
import json
from .dataset import Dataset


class NewsCategoryDataset(Dataset):
  name = "News Category Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    # Cmd that downloads the dataset
    # Works only when the dataset is a zipfile
    self.cmd = 'kaggle datasets download rmisra/news-category-dataset'

    # Download the dataset and extract to this folder
    self.dirname = 'news_category_dataset'

    # Absolute path to the above directory
    self.dataset_dir = None

    # Stores entire dataset (train+val+test) if any
    self.data = None

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
    with open(self.dataset_dir + '/News_Category_Dataset_v2.json', 'r') as f:
      self.data = [json.loads(line) for line in f.read().splitlines()]
    print("Done")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")

  
  # Split data
  def split_data(self):
    raise NotImplementedError("StanSent split_data not implemented!")
