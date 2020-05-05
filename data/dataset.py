import os
import glob
import subprocess
import zipfile


# Base Dataset class
class Dataset:
  # Constructor for class Dataset
  def __init__(self):
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


  def download_data(self, dirname, cmd):
    print("Downloading data to data/datasets/{0:} using the command:".format(dirname))
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

  # Get data as a tuple
  # Something like X_train, y_train, X_val, y_val
  def get_data(self):
    raise NotImplementedError("Base class method get_data not implemented!")

  def get_size(self):
    return len(self.data.index)

  # Split data
  def split_data(self, train_split):
    num_samples = len(self.data.index)
    num_train_samples = train_split*num_samples
    self.train_data = self.data.iloc[:num_train_samples, :]
    self.test_data = self.data.iloc[num_train_samples:, :]


  # Print string for class object
  def __str__(self):
    return self.__class__.name

