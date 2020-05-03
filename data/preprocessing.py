import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

# FIXME:
# Lemmatizer needs POS tags to work!

task_mapping_dict = {
  'tokenize': tokenize_text,
  'remove_punctuation': remove_punctuation,
  'remove_stopwords': remove_stopwords,
  'stemming': stem_text,
  'lemmatize': lemmatize_text
}

# For word_tokenize()
nltk.download('punkt')

# Stop words
stop_words = set(stopwords.words('english'))

# Remove punctuation
table = str.maketrans('', '', string.punctuation)

# Stemming
porter = PorterStemmer()

# Lemmatizing
lemmatizer = WordNetLemmatizer()


def preprocessor_fn(text, tasks=None):
  processed_text = text.copy()

  for task in tasks:
    processed_text = task_mapping_dict[task](processed_text)
  
  return processed_text


def tokenize_text(text):
  """
  Input: string
  Output: list of strings
  """
  return word_tokenize(text)


def remove_punctuation(words):
  """
  Input: list of strings
  Output: list of strings
  """
  return [w.translate(table) for w in words]


def remove_stopwords(words):
  """
  Input: list of strings
  Output: list of strings
  """
  tokens = [w for w in words if not w in stop_words]
  return tokens


def stem_text(words):
  """
  Input: list of strings
  Output: list of strings
  """ 
  stems = []
  for word in words:
    stems.append(porter.stem)
  return stems


def lemmatize_text(words):
  """
  Input: list of strings
  Output: list of strings
  """
  lemmatized_words = []
  for word in words:
    lemmatized_words.append(lemmatizer.lemmatize(word))
  return lemmatized_words
