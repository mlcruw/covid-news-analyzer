import os
import sys


class Config:
  # Define some class attributes
	## directory  
  # Please do not comment these lines as they store some important directories that will be used later on in the trainer object. These are global configurations.  
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  root_dir = os.path.join(cur_dir, '..')
  # datasets = ['stanford_sentiment']

  # change data directory
  # data_dir = os.path.join(root_dir, 'data/datasets')
  output_dir = os.path.join(root_dir, 'output')
  model_dir = os.path.join(output_dir, 'model_dump')
  # vis_dir = os.path.join(output_dir, 'vis')
  log_dir = os.path.join(output_dir, 'log')
  # result_dir = os.path.join(output_dir, 'result')

  ## model setting

  ## input, output

  ## training config

  ## testing config

  ## others
  #Add some default arguments so that creating a new object Config() below and in trainer.py would not prompt any error message.
  def __init__(self,
               dataset='news_cat',
               model='lr',
               feats='bow',
               save_path='bow_0',
               continue_train=False,
               load_path=None,
               test=None):
    self.dataset = dataset
    self.model = model
    self.feats = feats
    self.save_path = save_path
    self.continue_train = continue_train
    self.load_path = load_path
    self.test = test

#Do not comment "cfg" as it is a global configuration object that will be called later.
cfg = Config()
# sys.path.insert(0, cfg.root_dir)
# sys.path.insert(0, os.path.join(cfg.root_dir, 'data'))
from utils.dir_utils import add_pypath, make_folder

# add_pypath(os.path.join(cfg.data_dir))
# for i in range(len(cfg.datasets)):
#     add_pypath(os.path.join(cfg.data_dir, cfg.datasets[i]))

# Create the folder to store the best trained model
make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)

# Create the folder to store the log file
make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)
