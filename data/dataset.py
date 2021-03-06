import os
import glob
import subprocess
import zipfile
import pickle
import swifter
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess


# Base Dataset class
class Dataset:
  # Constructor for class Dataset
  def __init__(self, do_clean=True):
    print("Training on", self.__str__())

    # Cmd (string) that downloads the dataset
    # Works only when the dataset is a zipfile
    self.cmd = ''

    # Download the dataset and extract to this folder
    self.dirname = ''

    # Absolute path to the above directory
    self.dataset_dir = ''

    # Stores raw dataset (train+val+test)
    self.raw_data = None

    # Stores X and y (train+val+test)
    self.data = None

    # Stores label encoding
    self.label_encoder = None

    # Stores train data after train/test split
    self.train_data = None

    # Stores test data after train/test split
    self.test_data = None

    # Flag to determine whether to preprocess the data
    self.do_clean = do_clean


  def download_data(self, dirname, cmd):
    print("\nDownloading data to data/datasets/{0:} using the command:".format(dirname))
    print("  ", cmd)

    # Get the abs dir path of the current file dataset.py
    this_dir = os.path.dirname(__file__)

    # Create a folder 'datasets' inside current folder ('data')
    datasets_dir = os.path.join(this_dir, "datasets")
    if os.path.exists(datasets_dir) is False:
      os.mkdir(datasets_dir)

    # Create a folder with the given dirname inside 'datasets' directory
    curr_dataset_dir = os.path.join(datasets_dir, dirname)
    if os.path.exists(curr_dataset_dir) is False:
      os.mkdir(curr_dataset_dir)

    # If there are zip files inside this 'datasets' dir, delete them
    zip_files_in_datasets_dir = glob.glob(datasets_dir + '/*.zip')
    if zip_files_in_datasets_dir:
      for f in zip_files_in_datasets_dir:
        subprocess.run('rm '+ f,
                       shell=True,
                       cwd=datasets_dir)

    # If the current_dataset_dir is empty
    # implies dataset wasn't downloaded
    if not os.listdir(curr_dataset_dir):
      # Download dataset (assuming it's a zipfile)
      subprocess.run(cmd,
                     shell=True,
                     cwd=datasets_dir)

      downloaded_zipfile_path = glob.glob(datasets_dir + '/*.zip')[0]

      # Now extract the zipfile
      with zipfile.ZipFile(downloaded_zipfile_path, 'r') as z:
        z.extractall(curr_dataset_dir)

      # Now delete the zipfile
      subprocess.run("rm "+downloaded_zipfile_path,
                    shell=True,
                    cwd=this_dir)
    else:
      print("Not downloading. Data already downloaded")

    # Return abs path to where dataset was downloaded
    self.dataset_dir = curr_dataset_dir


  # Returns the size of the full dataset
  def get_size(self):
    return len(self.data.index)


  # If dataset if large, one may use dataset_ratio to use only a fraction
  # of the dataset
  def split_data(self, dataset_ratio=1.0, test_size=0.2):
    """
    Split the data into train and test splits

    Args:
      - dataset_ratio: ratio of dataset to use
      - test_size: proportion of data to use for testing
    """
    print("\nSplitting data...")
    data = self.data.iloc[:int(len(self.data.index)*dataset_ratio)]
    self.train_data, self.test_data = train_test_split(data, test_size=test_size, shuffle=False)
    print("Done")

  def preprocess_text(self, data):
    if self.do_clean:
      data['X'] = data['X'].swifter.apply(preprocess)
    return data

  # Print string for class object
  def __str__(self):
    return self.__class__.name


  # If labels were encoded, call this method
  # Saves label encoder object as a pickle file, inside dataset_dir
  def save_label_encoder(self):
    pickle.dump(self.label_encoder,
                open(self.dataset_dir + "/" + "label_encoder.pickle", "wb"))
