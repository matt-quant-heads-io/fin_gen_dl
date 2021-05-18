# ~~~~ machine learning experimentation libraries ~~~~
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN, LeakyReLU, Dropout, LSTM, Bidirectional
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
import time

import boto3
from io import StringIO



def create_train_test_set(df=None, train_split=None, scale_data=True, lookahead=1):
    # split the data into training and testing
    train_test_split = int(len(df) * train_split)

    # convert datetime from str to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # set datetime col to be index
    df.set_index('datetime', inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['datetime'], inplace=True)

    # use for hourly data to filter non-data at end (if not sorted)
    dfn = df.copy()

    # create y
    y = dfn['close']
    dfn.drop(columns=['close'], inplace=True)

    # create y-values-final
    y_tr = []
    for i in range(len(y) - lookahead + 1):
        push = y[i:i + lookahead]
        y_tr.append(push)

    y_tr = np.array(y_tr)

    # create numpy arrays
    Xtrain = np.array(dfn.iloc[:train_test_split, :])
    Xtest = np.array(dfn.iloc[train_test_split:, :])

    ytrain = y_tr[:train_test_split]
    ytest = y_tr[train_test_split:]

    if scale_data:
        # scale Xtrain, Xtest
        Xscaler = StandardScaler()  # scale so that all the X data will range from 0 to 1
        Xscaler.fit(Xtrain)
        Xtrain = Xscaler.transform(Xtrain)
        Xscaler.fit(Xtest)
        Xtest = Xscaler.transform(Xtest)

        # scale ytrain, ytest
        Yscaler = StandardScaler()
        Yscaler.fit(ytrain)
        ytrain = Yscaler.transform(ytrain)

    return Xtrain, Xtest, ytrain, ytest, train_test_split