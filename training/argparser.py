import argparse

# Parse command line args
def argparser():
  parser = argparse.ArgumentParser()

  parser.add_argument('-c',
                      '--continue_train',
                      action='store_true',
                      help='when set, training will continue from before')
  
  #error: ambiguous option: --test could match --test_only, --test_ratio
  parser.add_argument('-test_only',
                      dest='test_only',
                      action='store_true',
                      help='when set, testing will be done (we are in the TEST PHASE, no need to train)')

  parser.add_argument('--dataset',
                      default='news_cat',
                      type=str,
                      choices=['stan_sent', 'news_cat', 'fake_news', 'emo_aff'],
                      help='dataset to use',
                      dest='dataset')

  parser.add_argument('--models',
                      nargs='+',
                      type=str,
                      required=True,
                      help='model(s) to run')

  # TODO: Add choices?
  parser.add_argument('--feats',
                      nargs='+',
                      type=str,
                      help='features to use while training',
                      dest='feats')

  parser.add_argument('--save_path',
                      type=str,
                      help='path to where models would be saved',
                      dest='save_path')

  parser.add_argument('--load_path',
                      type=str,
                      help='provide path to where models were saved',
                      dest='load_path')

  parser.add_argument('--split_ratio',
                      type=float,
                      default=0.9,
                      help='train / val split ratio ($\in$ [0.0, 1.0])',
                      dest='split_ratio')
    
  #optional: dataset is too large; takes too long to train or test the whole (This is for debugging convenience)
  parser.add_argument('--test_ratio',
                      type=float,
                      default=0.0,
                      help='ratio to test, when set to 0.0 uses the above val split ratio otherwise use test_ratio * num of samples',
                      dest='test_ratio')

  args = parser.parse_args()

  return args
