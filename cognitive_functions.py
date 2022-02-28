import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
# from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("YOUR_TRANSLATOR_KEY")
endpoint = os.getenv("YOUR_TRANSLATOR_ENDPOINT")


# Authenticate the client using your key and endpoint
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint,
                                                credential=ta_credential)
    return text_analytics_client


client = authenticate_client()


# build a results dataframe
def build_results_dataframe():
    return pd.DataFrame(columns=['Tweet', 'Target', 'Prédit'])


results = build_results_dataframe()


# predict sentiment from tweets using Azure Text Analytics
def predict_sentiment(client, text):
    documents = [text]
    response = client.analyze_sentiment(documents=documents)[0]
    for index, sentence in enumerate(response.sentences):
        return sentence.sentiment


# predict sentiment from dataframe

# timer_end = 0


def predict_df_sentiment(df, results, text, target):
    '''
    df, results, text='text', target='target'
    '''
    timer_start = time()
    for index, row in df.iterrows():
        results = results.append(
            {
                'Tweet': row[text],
                'Target': row[target],
                'Prédit': predict_sentiment(client, row[text])
            },
            ignore_index=True)

    timer_end = time() - timer_start

    results = results[results['Prédit'] != 'neutral']
    results['Prédit'] = results['Prédit'].replace(['negative', 'positive'],
                                                  [0, 1])

    print(f'Le nombre de ligne perdu est de {len(df)-len(results)}\
        sur un total de {len(df)} lignes, soit {len(results)/len(df)*100}% le tout en {timer_end} secondes')


# # Statistics variables
# y_test = results.label.astype(int)
# y_pred = results.prediction

# # AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# auc_score = auc(fpr, tpr)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)

# save the score
score_cognitive = []
score_cognitive = pd.DataFrame.from_dict(score_cognitive)


def save_score_cognitive(df_score,
                         df_results,
                         auc_score,
                         accuracy,
                         timer_end,
                         model_name='Cognitive Service'):
    '''
    df_score=score_cognitive, df_results=results,  model_name='Cognitive Service'
    '''
    df_score.append({
        'Modèle': model_name,
        'Durée (s)': '{:0.1f}'.format(timer_end),
        'Nb de lignes': '{:0.1f}'.format(len(df_results)),
        'Score ROC (%)': '{:0.3f}'.format(auc_score * 100),
        'Accuracy (%)': '{:0.3f}'.format(accuracy * 100)
    })

    return df_score


def plot_confusion_matrix(y_test, y_pred, model_title="Tweets clean"):
    '''
    y_test=y_test, y_pred=y_pred, model_title="Tweets clean"
    '''

    # Create confusion matrix table
    cm = pd.crosstab(index=y_test,
                     columns=y_pred,
                     values=y_test,
                     aggfunc=lambda x: len(x),
                     normalize='index').mul(100)

    # Plot confusion matrix
    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    ax = sns.heatmap(cm, annot=True, fmt='.1f', cbar=False, cmap='Blues')

    for t in ax.texts:
        t.set_text(t.get_text() + " %")
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')

    plt.ylabel('Réelle', fontweight='bold')
    plt.xlabel('Estimée', fontweight='bold')

    title = 'Matrice de confusion - ' + model_title
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    cr = classification_report(y_test, y_pred)

    print(cr)