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
    print("Reading data...")
    self.train_data = pandas.read_csv(self.dataset_dir + "/train.tsv", sep="\t")
    self.test_data = pandas.read_csv(self.dataset_dir + "/test.tsv", sep="\t")
    print("Done")

  def split_data(self, **kwargs):
    pass

  # Standardize data
  def standardize_data(self):
    raise NotImplementedError("StanSent standardize_data method doesn't exist!")
