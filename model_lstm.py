# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:15:53 2020

@author: Afreen Kazi
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM



def LSTM_Model(step_size):
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, step_size), return_sequences = True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model
lstm = LSTM_Model(1)


def compile_and_fit(model, X_train, y_train, patience, epochs):
    import tensorflow as tf
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2)
    history = model.fit(X_train, y_train, epochs=epochs,
                      validation_split=0.2,
                      callbacks=[early_stopping])
    return history
history = compile_and_fit(lstm, train_X_r, train_Y, patience=5, epochs=5)

def predict(lstm, test_X, test_Y):
    from sklearn.metrics import mean_squared_error
    pred = lstm.predict(test_X)
    error = mean_squared_error(pred, test_Y)
    return pred, error

pred, error = predict(lstm, test_X_r, test_Y)
print(error)



predicted = predict(lstm, test_X_r, test_Y_r)

def plot_ts(actual_values, predicted_values):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(actual_values)
    plt.plot(predicted_values)
    plt.legend('actual', 'predicted')
    plt.title('Acutal vs predicted')
    plt.show()
    
plot_ts(test_Y, pred)


