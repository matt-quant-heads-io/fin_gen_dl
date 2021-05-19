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
from tensorflow.keras.layers import Dense, SimpleRNN, LeakyReLU, Dropout
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model

import boto3
from io import StringIO


import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


lookahead = 2
lookback = 2
batchsize = 200
numunits = 100
epochscount = 2


def create_train_test_set(df=None, train_split=None,scale_data=True):
    
    # split the data into training and testing
    train_test_split = int(len(df) * train_split)
    
    # convert datetime from str to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # set datetime col to be index
    df.set_index('datetime', inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['datetime'], inplace=True)
        
    # use for hourly data to filter non-data at end (if not sorted)
    dfn = df.iloc[:-907, :]
        
    # create y    
    y = dfn['close']
    dfn.drop(columns=['close'], inplace=True)
    
    # create y-values-final
    y_tr = []
    for i in range(len(y)-lookahead+1):
        push = y[i:i+lookahead]
        y_tr.append(push)
    
    y_tr = np.array(y_tr)
    
    # create numpy arrays
    Xtrain = np.array(dfn.iloc[:train_test_split, :])
    Xtest = np.array(dfn.iloc[train_test_split:, :])
        
    ytrain = y_tr[:train_test_split]
    ytest = y_tr[train_test_split:]

    if scale_data:
        # scale Xtrain, Xtest
        Xscaler = StandardScaler() # scale so that all the X data will range from 0 to 1
        Xscaler.fit(Xtrain)
        Xtrain = Xscaler.transform(Xtrain)
        Xscaler.fit(Xtest)
        Xtest = Xscaler.transform(Xtest)

        # scale ytrain, ytest
        Yscaler = StandardScaler()
        Yscaler.fit(ytrain)
        ytrain = Yscaler.transform(ytrain)
        
    return Xtrain, Xtest, ytrain, ytest, train_test_split


# note - update the batch size

def create_model(output_units=lookahead, lr=0.0001, loss='mse', num_units=numunits,
                 activation_func='relu', batch_size=batchsize, dropout=0.1,
                alpha=0.5, n_inputs=None, n_features=None,
                 optimizer='adam', show_model_summary=True):
    try:
        if n_inputs is None or n_features is None:
            raise("n_inputs and n_features cannot be None type")
        else:
            adam = Adam(lr=lr)

            # Initialize the RNN
            model = Sequential()
            model.add(SimpleRNN(units=num_units,
                           activation=activation_func,
                           input_shape=(n_inputs, n_features)))
            model.add(Dense(units=output_units, activation='linear'))
            
#             model = Sequential()
#             model.add(LSTM(units=num_units,
#                            activation=activation_func,
#                            input_shape=(n_inputs, n_features)))
#             model.add(LeakyReLU(alpha=alpha))
#             model.add(Dropout(dropout))
#             model.add(Dense(units=output_units))

            # Compiling the RNN
            model.compile(optimizer, loss)
            if show_model_summary:
                model.summary()

            return model
    except Exception as e:
        msg = "Error in create_model"
        raise e(msg)

# ~~~~ load and split the training and test data ~~~~

aws_id = ''
aws_secret = ''

client = boto3.client('s3', aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret)

bucket_name = ''
object_key = ''
csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')
data = pd.read_csv(StringIO(csv_string), header=0)

df = data


train_split_size = 0.8
Xtrain, Xtest, ytrain, ytest, train_test_split = create_train_test_set(df=df, train_split=train_split_size)


# for i in range(10):
#     x, y = generator[i]
# #     print(x.shape)
# #     print(y.shape)
#     print('%s => %s' % (x, y))
#     print(len(y))


# baseline

loss_res = np.zeros((10, epochscount))

for i in range(10):
    K.clear_session()
    print(i)

    # setup model & TimeseriesGenerator, and train the model
    n_inputs = lookback
    n_features = Xtrain.shape[1]

    generator = TimeseriesGenerator(Xtrain[0:2000], ytrain[0:2000], length=lookback, batch_size=1)
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    model.fit_generator(generator, epochs=epochscount)
    
    loss_val_per_epoch = model.history.history['loss']
    loss_res[i] = loss_val_per_epoch

pd.DataFrame(loss_res).to_csv("baseline.csv")


# update input/output parameters

loss_res = np.zeros((10, epochscount))
for i in range(0,10):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i,:] = loss_val_per_epoch
    
    print(loss_res)
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("0-10.csv")

loss_res = np.zeros((10, epochscount))
for i in range(10,20):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-10,:] = loss_val_per_epoch
    
    print(loss_res)
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("10-20.csv")


loss_res = np.zeros((10, epochscount))
for i in range(20,30):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-20,:] = loss_val_per_epoch
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("20-30.csv")


loss_res = np.zeros((10, epochscount))
for i in range(30,40):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-30,:] = loss_val_per_epoch
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("30-40.csv")


loss_res = np.zeros((10, epochscount))
for i in range(40,50):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-40,:] = loss_val_per_epoch
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("40-50.csv")


loss_res = np.zeros((10, epochscount))
for i in range(50,60):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-50,:] = loss_val_per_epoch
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("50-60.csv")


loss_res = np.zeros((10, epochscount))
for i in range(60,70):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-60,:] = loss_val_per_epoch
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("60-70.csv")


loss_res = np.zeros((10, epochscount))
for i in range(70,74):
    
    K.clear_session()
    
    start = datetime.datetime.now()
    print(start)
    print('')
    
    print(Xtrain.shape)
    print(i)
    Xtrain_new = np.delete(Xtrain, i, axis = 1)
    print(Xtrain_new.shape)

    n_inputs = lookback
    n_features = Xtrain_new.shape[1]
    
    # setup model & TimeseriesGenerator, and train the model
    model = create_model(n_inputs=n_inputs, n_features=n_features)
    generator = TimeseriesGenerator(Xtrain_new[0:2000], ytrain[0:2000], length=lookback, batch_size=lookahead)
    model.fit_generator(generator, epochs=epochscount)

    print('')
    end = datetime.datetime.now()
    print(end)
    
    loss_val_per_epoch = model.history.history['loss']
    
    loss_res[i-70,:] = loss_val_per_epoch
    print(np.sum(loss_res))

pd.DataFrame(loss_res).to_csv("70-74.csv")
