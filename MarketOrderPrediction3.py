import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#from keras.utils.io_utils import HDF5Matrix
from keras.layers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU, PReLU
#from keras.optimizers import SGD, Adam, RMSprop
#from keras.utils import np_utils
import keras.backend as K
import h5py

from itertools import product
#from functools import partial

import pickle
import sys
import os
import time
import pandas as pd
import numpy as np
import yaml

ticker = 'AAPL'
starttime = '34200000'
endtime = '57600000'
nlevels = '100'


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class EpochPrint(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        print("Starting training.")

    def on_batch_end(self, batch, logs={}):
        if batch % 1000 == 0:
            print("Batch: %d" % batch)

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def concat_dataframes(dates, model_type, lob_loc):
    print("model type: %s" % model_type)
    data_X = None
    data_Y = None
    lob_loc = lob_loc
    for day in dates:
        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
        if os.path.exists(lob_loc + fname):
            X,y = get_daily_data(day, "pandas", lob_loc)
        else:
            print("%s does not exist!" %(lob_loc + fname))
            continue

        if data_X is None:
            data_X = X
            data_Y = y
        else:
            data_X = pd.concat([data_X, X])
            data_Y = pd.concat([data_Y, y])

    if isinstance(data_X, pd.DataFrame):
        data_X, data_Y =  data_X.values, data_Y.values

    # Standardize variables
    data_X = data_X - data_X.mean(axis=0)
    std = (data_X ** 2).sum(axis=0)
    data_X /= std
    return data_X, data_Y

def get_daily_data(day, datatype, lobster_location):

    print(day)
    feat_cols = ["Type", "LastTradeDuration","LastTradeVolume"]
    ask_cols = ["ASKp" + lev.__str__() for lev in range(1,11)]
    ask_cols.extend(["ASKs" + lev.__str__() for lev in range(1,11)])
    bid_cols = ["BIDp" + lev.__str__() for lev in range(1,11)]
    bid_cols.extend(["BIDs" + lev.__str__() for lev in range(1,11)])
    feat_cols.extend(ask_cols)
    feat_cols.extend(bid_cols)

    fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
    Lobster = pickle.load(open(lobster_location + fname, "rb"))
    orderbook = Lobster.orderbook
    orderbook = orderbook.loc[:, feat_cols] # only keep selected features
    #orderbook.loc[:,[col for col in orderbook.columns if col != "LastTradeDuration"]] = \
    #    orderbook.loc[:,[col for col in orderbook.columns if col != "LastTradeDuration"]] - \
    #    orderbook.loc[:,[col for col in orderbook.columns if col != "LastTradeDuration"]].shift(1) # create dataframe
        # of feature changes
    #orderbook = orderbook.iloc[1:] # remove the first row which is nan due to shift

    df_feat = orderbook
    df_feat["m_order"] = 0
    df_feat.loc[df_feat.Type==4, "m_order"] = 1
    df_feat.loc[:,"m_order"] = df_feat.loc[:,"m_order"].shift(-1)
    df_feat = df_feat.iloc[:-1]

    X = df_feat.loc[:,[col for col in df_feat.columns if (col != "m_order")]]
    y = df_feat.loc[:,["m_order"]]
    del X["Type"]
    del X["LastTradeVolume"]

    if datatype == "pandas":
        return X, y
    elif datatype == "numpy":
        return X.values, y.values
    else:
        print("datatype must be either 'pandas' or 'numpy'.")
        return 0,0

def get_rolling_windows(size, dates, lob_loc, model_type='LSTM'):
    data_X, data_Y = concat_dataframes(dates, model_type, lob_loc)
    nobs = data_X.shape[0]
    nfeat = data_X.shape[1]
    window_data = np.zeros((nobs - size + 1, size, nfeat))

    for i in range(size-1, nobs):
        if i % 1000000 == 0:
            print("row ", i)
        if i == nobs - 1:
            window_data[i-size+1, :, :] = data_X[(i-size+1):,:]
        else:
            window_data[i-size+1, :, :] = data_X[(i-size+1):(i+1),:]

    return window_data, data_Y[size-1:,]

def BundleGenerator(files):
    counter = 0
    for fname in files:
        #fname = files[counter]
        data_bundle = pickle.load(open(fname, "rb"))
        X_train = data_bundle[0]
        y_train = data_bundle[1]
        #counter = (counter + 1) % len(files)
        yield (X_train, y_train)

def BatchGenerator(files, batch_size):
    counter = 0
    while True:
        fname = files[counter]
        print(fname)
        counter = (counter + 1) % len(files)
        data_bundle = pickle.load(open(fname, "rb"))
        X_train = data_bundle[0].astype(np.float32)
        y_train = data_bundle[1].astype(np.float32)
        y_train = y_train.flatten()
        for cbatch in range(0, X_train.shape[0], batch_size):
            yield (X_train[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)])


if __name__ == "__main__":

    if len(sys.argv) == 1:
        train_bundle_loc = "/Volumes/INTENSO/LOBSTER/data_bundles/bin_enc/"
        test_bundle_loc = "/Volumes/INTENSO/LOBSTER/data_bundles/bin_enc/test_bundles/"
        num_epoch = 1
        batch_size = 133
        nb_train_bundles = 1
        nb_test_bundles = 1
    else:
        try:
            print("Reading inputs...")
            num_epoch, train_bundle_loc, test_bundle_loc, nb_train_bundles, nb_test_bundles = sys.argv[1:]
        except IndexError:
            print("Usage: MarketOrderPrediction.py\n<num_epoch (int)>\n<training_bundles_location (str)>\n<test_bundle_location (str)>\n<nb_train_bundles (int)>\n<nb_test_bundles (int)>")
            sys.exit(1)

    print("Starting program...")
    nb_tsteps = 10
    nb_feats = 40
    batch_size = 120 # 133 because a training bundle has 59318 samples and this is divisible by 133
    nb_train_bundles = int(nb_train_bundles)
    nb_test_bundles = int(nb_test_bundles)
    num_epoch = int(num_epoch)
    model_type='LSTM'
    samples_per_epoch = 65280 * nb_train_bundles


    print("Compiling model...")
    model = Sequential()
    model.add(TimeDistributed(Dense(120, activation = 'linear'), input_shape=(nb_tsteps, nb_feats)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(LSTM(250, return_sequences=False, consume_less='gpu'))
    model.add(Dense(100, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    print("Starting training...")
    stime = time.time()
    class_weights = {0 : 1, 1 : 13}

    train_files = [train_bundle_loc + "bundle_" + cb.__str__() for cb in range(nb_train_bundles)]
    gen = BatchGenerator(files=train_files, batch_size=batch_size)
    History = model.fit_generator(gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epoch,verbose=1, class_weight=class_weights)
    etime = time.time()
    print("Training time: ", etime - stime)

    print("Testing model...")
    # Open test set
    test_files = [test_bundle_loc + "bundle_" + cb.__str__() for cb in range(nb_test_bundles)]
    test_gen = BatchGenerator(files=test_files, batch_size=139) # dont change batch_size
    score = model.evaluate_generator(test_gen, val_samples=79091 * nb_test_bundles) # dont change val_samples
    print(score)

    print("Saving model...")
    model_json = model.to_json()
    with open("model_LSTM_v2.json", "w") as jf:
        jf.write(model_json)

    model.save_weights("modelweights_LSTM_v2.h5", overwrite=True)

    hist = History.history
    pickle.dump(hist, open("history_LSTM_v2.p", "wb"))


