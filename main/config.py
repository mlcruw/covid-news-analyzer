import os
import os.path as osp
import sys


class Config:
	## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    trainset = ['stanford_sentiment'] 
    #testset = []
    
    # change data directory
    data_dir = osp.join(root_dir, 'data/datasets')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
 
    ## model setting
    
    ## input, output

    ## training config

    ## testing config

    ## others
    
    def set_args(self, dataset='stanford_sentiment', model=['linearsvm'], feat='BoW'):
        self.trainset = [dataset] #TO-DO: when input is a list?
        self.model = model
        self.feat = feat
        


cfg = Config()
sys.path.insert(0, cfg.root_dir)
sys.path.insert(0, osp.join(cfg.root_dir, 'data'))
from utils.dir_utils import add_pypath, make_folder

add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset[i]))
#add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)