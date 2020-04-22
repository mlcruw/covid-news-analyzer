# DeepSentiment

The repository contains an LSTM deep learning network trained for a 10 class sentiment classification.

Please see the blog for more details on the model: https://medium.com/@prajwalshreyas/sentiment-analysis-for-text-with-deep-learning-2f0a0c6472b5

----
## Installation
1. pip3 install tensorflow
2. pip3 install keras

Make sure the versions of tf and keras match.


----
## Test
- Test one sentence (56 words max)
Change the string **data_samples** in **pred.py**.

```
python3 prediction_code/pred.py
```

----
## Train
- Train from scratch  
```
python3 train_code/sentiment_rnn_train.py
```
----
## Data & Models
Original: [https://gitlab.com/praj88/deepsentiment/-/tree/master/Data](https://gitlab.com/praj88/deepsentiment/-/tree/master/Data)

Modified: Go to Google Drive.
