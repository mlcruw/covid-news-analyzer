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
  def __init__(self, do_clean=True):
    super().__init__(do_clean)
    self.cmd = 'kaggle datasets download rmisra/news-category-dataset'
    self.dirname = 'news_category_dataset'

    # Download the dataset to dirname using cmd
    self.download_data()

    # Read data from dirname
    # Note: this is different for different datasets
    # So make sure to change for each dataset
    self.read_data()

    # self.standardize_data()

    self.save_label_encoder()


  # Download the dataset
  def download_data(self):
    super().download_data(self.dirname, self.cmd)


  # Read data from files
  def read_data(self):
    print("\nReading data...")
    with open(self.dataset_dir + '/News_Category_Dataset_v2.json', 'r') as f:
      # Read raw data into pandas dataframe
      raw_data = pd.read_json(f, lines=True)

      # Filter useful categories
      categories = ['POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'WELLNESS', 'BUSINESS',
              'SPORTS', 'SCIENCE', 'COMEDY', 'CRIME', 'RELIGION']
      raw_data = raw_data[raw_data['category'].isin(categories)]
      self.raw_data = raw_data

      unique_categories = list(pd.unique(raw_data.category))
      le = LabelEncoder()
      le.fit(unique_categories)
      self.label_encoder = le

      # Create a new pandas dataframe from raw_data columns
      data = pd.DataFrame()
      # Could we insert one space between the title and text?
      data['X'] = raw_data.headline + " " + raw_data.short_description

      if self.do_clean:
        # Preprocess the text
        data = self.preprocess_text(data)

      data['y'] = le.transform(raw_data.category)
      self.data = data

    print("Done")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")
