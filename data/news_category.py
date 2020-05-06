import glob
import zipfile
import pandas as pd
import json
from .dataset import Dataset
from sklearn.preprocessing import LabelEncoder

# TODO:
# - figure out a good split fraction
# - headline+short_description is what we're supposed to do?
# -

class NewsCategoryDataset(Dataset):
  name = "News Category Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    self.cmd = 'kaggle datasets download rmisra/news-category-dataset'
    self.dirname = 'news_category_dataset'

    # Download the dataset to dirname using cmd
    self.download_data()

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
    with open(self.dataset_dir + '/News_Category_Dataset_v2.json', 'r') as f:
      # Read raw data into pandas dataframe
      raw_data = pd.read_json(f, lines=True)

      # Read into a list of JSONs
      # raw_data = [json.loads(line) for line in f.read().splitlines()]

      self.raw_data = raw_data

      unique_categories = list(pd.unique(raw_data.category))
      le = LabelEncoder()
      le.fit(unique_categories)
      self.label_encoder = le

      # Create a new pandas dataframe from raw_data columns
      data = pd.DataFrame()
      data['X'] = raw_data.headline + raw_data.short_description
      data['y'] = le.transform(raw_data.category)
      self.data = data

    print("Done")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")


  # Split data
  def split_data(self, train_size=0.7, test_size=0.2):
    super().split_data(train_size, test_size)
