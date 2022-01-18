import itertools
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-cased")


def plot_model(history_model):
    # graphique des performances du modèle
    acc = history_model.history['acc']
    val_acc = history_model.history['val_acc']
    loss = history_model.history['loss']
    val_loss = history_model.history['val_loss']
    auc = history_model.history['auc']
    val_auc = history_model.history['val_auc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label="Accuracy de l'entraînement")
    plt.plot(epochs, val_acc, 'b', label="Accuracy de la validation")
    plt.title("Accuracy de l'entraînement et de la validation")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label="Perte à l'entraînement")
    plt.plot(epochs, val_loss, 'b', label="Perte à la validation")
    plt.title("Perte à l'entraînement et à la validation")
    plt.legend()
    plt.show()

    plt.plot(epochs, auc, 'bo', label="AUC à l'entraînement")
    plt.plot(epochs, val_auc, 'b', label="AUC à la validation")
    plt.title("AUC à l'entraînement et à la validation")
    plt.legend()
    plt.show()


def bertify_data(data, labels):
    Xids = np.zeros((len(data), 128))
    Xmask = np.zeros((len(data), 128))
    Y = pd.get_dummies(labels).values

    for i, sentence in enumerate(data):
        Xids[i, :], Xmask[i, :] = tokenize(sentence)

    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, Y))

    dataset = dataset.map(map_func)

    return dataset.shuffle(10000).batch(32)


def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


def tokenize(sentence):
    tokens = tokenizer_bert.encode_plus(sentence,
                                        max_length=128,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        return_attention_mask=True,
                                        return_token_type_ids=False,
                                        return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']


def word2vec(X,
             W2V_SIZE=300,
             W2V_WINDOW=7,
             W2V_EPOCH=32,
             W2V_MIN_COUNT=10,
             SEQUENCE_LENGTH=35,
             workers=32):
    documents = [_text.split() for _text in X]
    w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=workers)
    w2v_model.build_vocab(documents)

    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv.index_to_key:
            embedding_matrix[i] = w2v_model.wv.get_vector(word, norm=True)

    embedding_layer_word_2_vec = Embedding(vocab_size,
                                           W2V_SIZE,
                                           weights=[embedding_matrix],
                                           input_length=SEQUENCE_LENGTH,
                                           trainable=False)

    return embedding_layer_word_2_vec

def plot_confusion_matrix(model, x_test, y_test, model_title='Modèle'):
    y_pred = (model.predict(x_test).ravel() > 0.5) + 0
    classes = ['negative', 'positive']
    model_title = 'Confusion matrix - ' + model_title
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.text(-0.3, -0.7, model_title, fontweight='bold')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print(cr)