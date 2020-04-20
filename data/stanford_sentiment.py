import glob
import zipfile
import pandas
from .dataset import Dataset


class StanfordSentimentDataset(Dataset):
  name = "Stanford Sentiment Analysis Dataset"

  # class constructor
  def __init__(self):
    super().__init__()
    self.cmd = 'kaggle competitions download -c sentiment-analysis-on-movie-reviews'
    self.download_dirname = 'sentiment-analysis-on-movie-reviews'
    self.dirname = 'stanford_sentiment'
    self.download_data(self.dirname, self.cmd)
    self.read_data()
    # self.standardize_data()

  
  # Override base class download_data()
  def download_data(self, dirname, cmd):
    # absolute path to self.dirname
    self.dataset_dir = super().download_data(dirname, cmd)
    
    # further extract zip files inside extracted directory
    downloaded_zipfiles = glob.glob(self.dataset_dir + '/*.zip')
    for ff in downloaded_zipfiles:
      with zipfile.ZipFile(ff, 'r') as z:
        z.extractall(self.dataset_dir)
  
  
  # Read data from files
  def read_data(self):
    self.train_data = pandas.read_csv(self.dataset_dir + "/train.tsv", sep="\t")
    self.test_data = pandas.read_csv(self.dataset_dir + "/test.tsv", sep="\t")


  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")

  
  # Split data
  def split_data(self):
    raise NotImplementedError("StanSent split_data not implemented!")
