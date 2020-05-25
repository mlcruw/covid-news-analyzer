import random
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.linear_model import LogisticRegression
from newspaper import Article
from web.util import get_article, TRUSTED_SOURCES

import sys
BASE_PATH = ''
from evaluate import evaluate_all, load_models

app = Flask(__name__)
load_models(BASE_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Uses the models to make the predictions and displays results on the page
    """
    article = get_article(request.form['article_url'])

    if article is None:
        return "The article URL is invalid/not supported. " + \
            "Try one of the following news sources: {}".format(TRUSTED_SOURCES)

    predictions = evaluate_all(article.text, article.title, BASE_PATH, True)
    sentiment = predictions['sentiment']
    fake = predictions['fake']
    category = predictions['category']
    emotion = predictions['emotion']

    sentiment = '{:.1f}% positive'.format(float(sentiment)*100/4)

    return render_template('index.html', sentiment=sentiment, fake=fake, emotion=emotion, category=category)

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, port=8001)
