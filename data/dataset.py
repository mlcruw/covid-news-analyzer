import os
import glob
import subprocess
import zipfile


# Base Dataset class
class Dataset:
  def __init__(self):
    # List of both train and test articles/reviews 
    self.sentences = []

    # Necessary? Consider modifying these 2
    self.words = []
    self.labels = []
    
  def download_data(self, dirname, cmd):
    print("Downloading data to data/datasets/{0:} using the command:".format(dirname))
    print("  ", cmd)

    # Get the abs dir path of the current file dataset.py
    this_dir = os.path.dirname(__file__) 

    datasets_dir = os.path.join(this_dir, "datasets")

    if os.path.exists(datasets_dir) is False:
      os.mkdir(datasets_dir)

    curr_dataset_dir = os.path.join(datasets_dir, dirname)

    if os.path.exists(curr_dataset_dir) is False:
      os.mkdir(curr_dataset_dir)
    
    if not os.listdir(curr_dataset_dir):
      # Directory is empty ==> dataset was not downloaded

      # Download dataset (assuming it's a zip file)
      subprocess.run(cmd,
                     shell=True,
                     cwd=datasets_dir)

      downloaded_zipfile_path = glob.glob(datasets_dir + '/*.zip')[0]
      
      # Now extract the zip file
      with zipfile.ZipFile(downloaded_zipfile_path, 'r') as z:
        z.extractall(curr_dataset_dir)
    
    # Return abs path to where dataset was downloaded
    return curr_dataset_dir
      
  # Get data as a tuple
  # Something like X_train, y_train, X_val, y_val  
  def get_data(self):
    raise NotImplementedError("Base class method get_data not implemented!")

  def get_size(self):
    raise NotImplementedError("Base class method get_size not implemented!")
  