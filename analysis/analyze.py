from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np


def compute_accuracy_scores():
    df_true = pd.read_csv("true_results.csv")
    df_pred = pd.read_csv("pred_results.csv")

    df_true["Fakeness (0,1)"] = df_true["Fakeness (0,1)"].astype(int)
    df_pred["Fakeness (0,1)"] = df_pred["Fakeness (0,1)"].astype(int)

    fakeness_acc = accuracy_score(df_true["Fakeness (0,1)"].to_numpy(),
                                  df_pred["Fakeness (0,1)"].to_numpy())

    df_true["Sentiments (0-4)"] = df_true["Sentiments (0-4)"].astype(float)
    df_pred["Sentiments (0-4)"] = df_pred["Sentiments (0-4)"].astype(float)

    sentiment_acc = mean_squared_error(df_true["Sentiments (0-4)"].to_numpy(),
                                       df_pred["Sentiments (0-4)"].to_numpy())

    emo_count = 0
    for idx in range(len(df_true["Emotion 1-7"])):
        emo_true = df_true["Emotion 1-7"].iloc[idx]
        emo_pred = df_pred["Emotion 1-7"].iloc[idx]

        if emo_true.lower() in emo_pred.lower():
            emo_count += 1

    emotion_acc = emo_count / len(df_true["Emotion 1-7"])

    cat_count = 0
    for idx in range(len(df_true["Categories"])):
        cat_true = df_true["Categories"].iloc[idx]
        cat_pred = df_pred["Categories"].iloc[idx]

        if cat_true == cat_pred:
            cat_count += 1

    categories_acc = cat_count / len(df_true["Categories"])

    data = {"Fakeness":   [fakeness_acc],
            "Sentiment":  [sentiment_acc],
            "Emotion":    [emotion_acc],
            "Categories": [categories_acc]}
    df_out = pd.DataFrame(data)
    print(df_out.to_latex())


if __name__=="__main__":
    compute_accuracy_scores()