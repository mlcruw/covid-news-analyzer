import csv
import pandas as pd
from newspaper import Article
from newspaper import fulltext

urls = ['https://www.scmp.com/news/world/europe/article/3047848/china-coronavirus-germany-confirms-first-case']
# urls= \
# ['https://www.foxnews.com/politics/coronavirus-wuhan-lab-china-compete-us-sources',
# 'https://www.foxnews.com/us/california-protest-erupts-over-states-coronavirus-stay-at-home-rules',
# 'https://www.foxnews.com/us/president-trump-speaks-coronavirus',
# 'https://www.foxnews.com/health/what-you-should-know-about-the-coronavirus-outbreak',
# 'https://www.foxnews.com/health/army-ebola-treatment-coronavirus-vaccine-sought',
# 'https://www.nytimes.com/2020/01/18/world/asia/china-virus-wuhan-coronavirus.html',
# 'https://www.nytimes.com/2020/01/08/health/china-pneumonia-outbreak-virus.html',
# 'https://www.nytimes.com/2020/02/09/world/asia/china-coronavirus-tests.html',
# 'https://www.nytimes.com/2020/02/09/world/asia/coronavirus-family-china.html',
# 'https://www.nytimes.com/2020/03/09/upshot/coronavirus-oil-prices-bond-yields-recession.html',
# 'https://www.cnn.com/2020/04/15/politics/kellyanne-conway-covid-19-coronavirus/index.html',
# 'https://www.cnn.com/2020/04/19/politics/trump-briefing-sunday-april-19/index.html',
# 'https://www.cnn.com/world/live-news/coronavirus-outbreak-03-14-20-intl-hnk/h_5cf82c46305a52487b6716a9bdcc80a3',
# 'https://www.cnn.com/asia/live-news/coronavirus-outbreak-02-12-20-intl-hnk/h_564f37dcc11578a0682f1783461c0c9d',
# 'https://www.cnn.com/asia/live-news/coronavirus-outbreak-02-12-20-intl-hnk/h_b416f227bc3fd3c61f33ea82e1d91170',
# 'https://www.scmp.com/news/world/europe/article/3047848/china-coronavirus-germany-confirms-first-case',
# 'https://www.scmp.com/news/world/united-states-canada/article/3047462/china-coronavirus-us-investigating-second-suspected',
# 'https://www.scmp.com/economy/china-economy/article/3075133/chinas-inbound-foreign-direct-investment-plunges-february',
# 'https://www.scmp.com/news/china/science/article/3079879/chinas-initial-coronavirus-outbreak-wuhan-spread-twice-fast-we',
# 'https://www.scmp.com/economy/china-economy/article/3077760/coronavirus-chinas-march-pmi-steadies-economy-not-out-woods']


df = pd.DataFrame(columns=['author','publish_date', 'title', 'text', 'source', 'url'])
for idx,url in enumerate(urls):
    if url:
        a_row = []
        article = Article(url)

        article.download()
        article.parse()

        if idx == 0:
            article.publish_date = "2020-04-15"
        if idx == 1:
            article.publish_date = "2020-04-19"
        if idx == 2:
            article.publish_date = "2020-01-31"
        if idx == 3:
            article.publish_date = "2020-01-31"
        if idx == 4:
            article.publish_date = "2020-03-30"

        # Just take the first author since
        # some articles tag on more than needed.
        a_row.append(article.authors[0])
        if type(article.publish_date) != str:
            a_row.append(article.publish_date.date())
        else:
            a_row.append(article.publish_date)
        a_row.append(article.title)
        a_row.append(fulltext(article.html))
        source = url.split('.')[1]
        a_row.append(source)
        a_row.append(url)

        df.loc[idx] = a_row

print(df)
df.to_csv("covid_19_articles.csv", index=False)