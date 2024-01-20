import re
import matplotlib as mpl
import pandas as pd
from flask import render_template, request, redirect, flash
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from app.page import bp
from flask import Flask, render_template, request, render_template, jsonify, redirect
import numpy as np
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import model_from_json

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from joblib import load
import re
from pymystem3 import Mystem
import pickle
import json
from keras.preprocessing.text import Tokenizer



@bp.route('/')
def index():
    return render_template("page/index.html")


@bp.route('/', methods=['GET', 'POST'])
def lern():
    index = request.form.get("index")
    if index=='1':

        df = pd.read_csv('app/page/labeled.csv')
        df['comment'] = df['comment'].apply(lambda x: text_cleaner(x))
        df = df.drop(df[df['comment'] == ''].index)
        comments = df['comment'].to_numpy()
        lemmatizator = Mystem()
        text_for_lemmatization = ' sep '.join(comments)
        lemmatizated_text = lemmatizator.lemmatize(text_for_lemmatization)
        lemmatizated_text_list = [word for word in lemmatizated_text if word != ' ' and word != '-']
        lemmatizated_text = ' '.join(lemmatizated_text_list)
        lemmatizated_array = np.asarray(lemmatizated_text.split(' sep '))
        df['toxic'] = df['toxic'].astype(int)
        labels = df['toxic'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(lemmatizated_array, labels, test_size=0.2, stratify=labels,
                                                            shuffle=True, random_state=42)
        token_counts = Counter()
        for sent in X_train:
            token_counts.update(sent.split(' '))

        dict_size = len(token_counts.keys())
        tokenizer = Tokenizer(num_words=dict_size)
        tokenizer.fit_on_texts(X_train)
        X_train_tokenized = tokenizer.texts_to_sequences(X_train)
        X_test_tokenized = tokenizer.texts_to_sequences(X_test)
        max_comment_length = 250
        X_train_padded = pad_sequences(X_train_tokenized, maxlen=max_comment_length)
        X_test_padded = pad_sequences(X_test_tokenized, maxlen=max_comment_length)
        max_features = dict_size
        embedding_dim = 64

        model = Sequential()
        model.add(Embedding(input_dim=max_features,
                            output_dim=embedding_dim,
                            input_length=max_comment_length))
        model.add(Conv1D(filters=embedding_dim * 2,
                         kernel_size=2,
                         activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        epochs = 5

        history = model.fit(X_train_padded, y_train, epochs=epochs, validation_data=(X_test_padded, y_test),
                            batch_size=512)
        mpl.use('TkAgg')
        plt.figure(figsize=(10, 10))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('app/static/media/foo.png')

        dump(model, 'app/page/modelka.joblib')
        flash('Модель обучена')
        return redirect(request.url)
    else:

        # Load the saved tokenizer object from a file
        with open('app/page/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Load the saved word index from a file
        with open('app/page/word_index.json', 'r') as handle:
            word_index = json.load(handle)
        lemmatizator = Mystem()
        dict_size = 28064

        max_comment_length = 250

        model = load('app/page/modelka.joblib')
        model.load_weights('app/page/model.h5')
        phrase = request.form.get("phrase")

        clean_example = text_cleaner(phrase)
        lemm_example = ' '.join(lemmatizator.lemmatize(clean_example))
        array_example = np.array([lemm_example])
        seq_example = tokenizer.texts_to_sequences(array_example)
        max_comment_length = 250
        pad_example = pad_sequences(seq_example, maxlen=max_comment_length)
        pred_example = model.predict(pad_example)
        if pred_example[0][0] >= 0.5:
            res = 'toxic'
            pr = round(pred_example[0][0] * 100, 4)
        else:
            res = 'not toxic'
            pr = 100 - round(pred_example[0][0] * 100, 4)
        s = res + ' с точностью ' + str(pr) + '%'
        flash(s)
        return redirect(request.url)
def text_cleaner(text):
    tokenized_text = word_tokenize(text, language='russian')
    clean_text = [word.lower() for word in tokenized_text if word not in punctuation and word != '\n']
    r = re.compile("[а-яА-Я]+")
    russian_text = ' '.join([w for w in filter(r.match, clean_text)])
    return russian_text



