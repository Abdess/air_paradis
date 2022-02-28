import json
import pickle5 as pickle
from time import time
import os
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def init():
    global model_lstm
    global tokenizer

    # Obtenir le chemin où le modèle déployé peut être trouvé
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'deploiement')

    # chargement du modèle
    model_lstm = load_model(model_path + '/lstm_glove.h5')

    # chargement du tokéniseur
    with open(model_path + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


# Handle request to the service
def run(data):
    try:
        # Récupérer la propriété textuelle de la requête JSON
        # Détails JSON attendus {"text" : "un texte à prédire pour le sentiment"}
        data = json.loads(data)
        prediction = predict(data['text'])
        return prediction
    except Exception as e:
        error = str(e)
        return error


# décoder la prédiction de score du modèle, pour qu'elle soit 0 ou 1
def decode_prediction(prediction):
    return 'Négatif' if prediction < 0.5 else 'Positif'


def clean_text(tweet, stopwords=False):
    tweet_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    tweet = re.sub(tweet_cleaning_re, ' ', str(tweet).lower()).strip()

    tokens = tweet.split()

    if stopwords is False:
        tokens = [w for w in tokens]
    else:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)


# Prédire le sentiment à l'aide du modèle
def predict(input_text):

    start = time()

    # nettoyage du texte de la requête
    input_text = clean_text(input_text)

    # tokéniser et rembourrer la requête test comme dans l'entraînement
    input_text = pad_sequences(tokenizer.texts_to_sequences([input_text]),
                               maxlen=35)

    # obtenir la prédiction du modèle
    prediction = model_lstm.predict([input_text])[0]

    # obtenir la prédiction de décodage
    label = decode_prediction(prediction)

    return {
        'Label': label,
        'Score': float(prediction),
        'Temps': time() - start
    }