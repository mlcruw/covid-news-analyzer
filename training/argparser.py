import argparse

# Parse command line args
def argparser():
  parser = argparse.ArgumentParser()

  parser.add_argument('-c',
                      '--continue_train',
                      action='store_true',
                      help='when set, training will continue from before')

  parser.add_argument('-t',
                      '--test',
                      action='store_true',
                      help='when set, testing will be done')

  parser.add_argument('--dataset',
                      default='news_cat',
                      type=str,
                      choices=['stan_sent', 'news_cat', 'fake_news'],
                      help='dataset to use',
                      dest='dataset')

  parser.add_argument('--models',
                      nargs='+',
                      type=str,
                      required=True,
                      help='model(s) to run')

  # TODO: Add choices?
  parser.add_argument('--feat',
                      type=str,
                      help='features to use while training',
                      dest='feat')

  parser.add_argument('--save_path',
                      type=str,
                      help='path to where models would be saved',
                      dest='save_path')

  parser.add_argument('--load_path',
                      type=str,
                      help='provide path to where models were saved',
                      dest='load_path')

  args = parser.parse_args()

  return args
