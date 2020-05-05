import os
import sys


class Config:
  # Define some class attributes
	## directory
  # cur_dir = os.path.dirname(os.path.abspath(__file__))
  # root_dir = os.path.join(cur_dir, '..')
  # datasets = ['stanford_sentiment']

  # change data directory
  # data_dir = os.path.join(root_dir, 'data/datasets')
  # output_dir = os.path.join(root_dir, 'output')
  # model_dir = os.path.join(output_dir, 'model_dump')
  # vis_dir = os.path.join(output_dir, 'vis')
  # log_dir = os.path.join(output_dir, 'log')
  # result_dir = os.path.join(output_dir, 'result')

  ## model setting

  ## input, output

  ## training config

  ## testing config

  ## others
  def __init__(self,
               dataset,
               model,
               feat,
               save_path,
               continue_train=False,
               load_path=None,
               test=None):
    self.dataset = dataset
    self.model = model
    self.feat = feat
    self.save_path = save_path
    self.continue_train = continue_train
    self.load_path = load_path
    self.testing = test

# cfg = Config()
# sys.path.insert(0, cfg.root_dir)
# sys.path.insert(0, os.path.join(cfg.root_dir, 'data'))
# from utils.dir_utils import add_pypath, make_folder

# add_pypath(os.path.join(cfg.data_dir))
# for i in range(len(cfg.datasets)):
#     add_pypath(os.path.join(cfg.data_dir, cfg.datasets[i]))
# make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)
# make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)
