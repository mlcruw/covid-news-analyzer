import random
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
model = LogisticRegression()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Uses the models to make the predictions and displays results on the page
    """
    print(request.form.values)
    # prediction = model.predict(final_features)

    categories = ['Science', 'Politics', 'Religion', 'Entertainment', 'Medicine']
    emotions = ['Sad', 'Happy', 'Angry', 'Excited', 'Worried']

    sentiment = '{}% positive'.format(random.randint(0,100))
    if random.random() > 0.5:
        fake = 'Fake'
    else:
        fake = 'Genuine'
    emotion = emotions[random.randint(0, len(emotions)-1)]
    category = categories[random.randint(0, len(categories)-1)]

    return render_template('index.html', sentiment=sentiment, fake=fake, emotion=emotion, category=category)

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, port=8001)
