"""
    Util file for processing a request from the site.
"""

from newspaper import Article
from urllib.parse import urlparse


# Set of new sources the ML models have been trained on.
TRUSTED_SOURCES = {"cnn",
                   "npr",
                   "nytimes",
                   "vox",
                   "wired"}


def get_article(URL):
    """
    Get an article from one our trusted sources.

    Args:
        URL: URL string to parse, e.g., http://www.hello.com/world

    Returns
        Article object if URL was success requested and parsed.
        None if it fails to parse or the URL is from a source not
        in the trusted list.
    """
    try:
        output = urlparse(URL)
        source = output.netloc.split('.')[1]
    except:
        print("Failed to parse URL.")
        return None

    if source not in TRUSTED_SOURCES:
        print("URL isn't in TRUSTED_SOURCES")
        return None

    article = Article(URL)
    article.download()
    article.parse()

    return article


# Example Code.
# if __name__=="__main__":
#     article = get_article("https://www.cnn.com/2020/05/02/investing/warren-buffett-berkshire-hathaway-earnings/index.html")

#     if article is not None:
#         print(article.text)
#         print(article.authors)