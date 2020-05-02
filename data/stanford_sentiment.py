import glob
import zipfile
import pandas
from .dataset import Dataset


class StanfordSentimentDataset(Dataset):
  name = "Stanford Sentiment Analysis Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    # Cmd that downloads the dataset
    # Works only when the dataset is a zipfile
    self.cmd = 'kaggle competitions download -c sentiment-analysis-on-movie-reviews'

    # Download the dataset and extract to this folder
    self.dirname = 'stanford_sentiment'

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
    
    # Further extract zip files inside extracted directory
    downloaded_zipfiles = glob.glob(self.dataset_dir + '/*.zip')
    for ff in downloaded_zipfiles:
      with zipfile.ZipFile(ff, 'r') as z:
        z.extractall(self.dataset_dir)
  
  
  # Read data from files
  def read_data(self):
    print("Reading data...")
    self.train_data = pandas.read_csv(self.dataset_dir + "/train.tsv", sep="\t")
    self.test_data = pandas.read_csv(self.dataset_dir + "/test.tsv", sep="\t")
    print("Done")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")

  
  # Split data
  def split_data(self):
    raise NotImplementedError("StanSent split_data not implemented!")
