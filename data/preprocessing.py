import string
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# FIXME:
# Lemmatizer needs POS tags to work!

# For word_tokenize()
nltk.download('punkt')

# Class for lemmatizing
# Does lemmatization after tokenizing using NLTK's tokenizer
class LemmaTokenizer:
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    return ' '.join([self.wnl.lemmatize(t) for t in word_tokenize(doc)])


# Class for stemming operation
# Does stemming after using NLTK's tokenizer
class StemTokenizer:
  def __init__(self):
    self.porter = PorterStemmer()
  def __call__(self, doc):
    return ' '.join([self.porter.stem(t) for t in word_tokenize(doc)])


lemmatize = LemmaTokenizer()
stem = StemTokenizer()
def preprocess(x, do_stem=False, do_lemmatize=True):
  """
  Final preprocessing method used to preprocess datasets
  """
  x = re.sub('[^a-z\s]', '', x.lower()) # convert to lowercase and remove noise
  if do_stem:
    x = stem(x.lower())
  if do_lemmatize:
    x = lemmatize(x)
  return x

# def tokenize_text(text):
#   """
#   Input: string
#   Output: list of strings
#   """
#   return word_tokenize(text)


# def remove_punctuation(words):
#   """
#   Input: list of strings
#   Output: list of strings
#   """
#   return [w.translate(table) for w in words]


# def remove_stopwords(words):
#     """
#     Input: list of strings
#     Output: list of strings
#     """
#     tokens = [w for w in words if not w in stop_words]
#     return tokens


# def stem_text(words):
#     """
#     Input: list of strings
#     Output: list of strings
#     """
#     stems = []
#     for word in words:
#         stems.append(porter.stem)
#     return stems


# def lemmatize_text(words):
#     """
#     Input: list of strings
#     Output: list of strings
#     """
#     lemmatized_words = []
#     for word in words:
#         lemmatized_words.append(lemmatizer.lemmatize(word))
#     return lemmatized_words


# #Fix the order otherwise prompt says it cannot find tokenize_text ...
# task_mapping_dict = {
#   'tokenize': tokenize_text,
#   'remove_punctuation': remove_punctuation,
#   'remove_stopwords': remove_stopwords,
#   'stemming': stem_text,
#   'lemmatize': lemmatize_text
# }


# def preprocessor_fn(text, tasks=None):
#     processed_text = text#.copy() string is immutable

#     for task in tasks:
#         processed_text = task_mapping_dict[task](processed_text)

#     return processed_text

