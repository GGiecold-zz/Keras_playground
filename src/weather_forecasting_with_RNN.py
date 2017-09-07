#!/usr/bin/env python


r"""Predicting daily temperatures from temperature, atmospheric pressure
    and hygrometry time series.
    This problem is meant to illustrate the use of recurrent neural networks
    (namely LSTM and GRU) and of various afferent techniques, such as
    recurrent dropout, stacking of recurrent layers, and bidirectional
    recurrent layers to better address the vanishing gradient problem
    and forgetting of sequential information.

    Instead of gathering all daily time series into a single large
    array, careful consideration is given to the possibility of
    duplicated rows, missing measurements, etc.
    This is accomplished by massaging the data into a multi-indexed
    dataframe (in the case at hand a two-level index accessed by
    specifying a day and a timestamp).
    Indeed, it turns out that there are 287 timestamps on
    '2010-01-07', instead of the expected 144 (on most days measurements
    are recorded every 10 minutes).
    As another illustration, '2016-10-25' comes with only 64 timestamps; 
    for some unaccounted reason, measurements range only from 12:00:00 A.M.
    to 10:30:00 A.M. on that day.

    A related issue addressed in the code herewith is that of
    calendar gaps. This would be overlooked by simply aggregating
    all time series in an array and would affect the purpose
    of making forecasts one or more days ahead.
"""


from __future__ import print_function

from builtins import range, zip
import errno
from inspect import getargspec, isgenerator

try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

try:
    from multiprocessing import cpu_count
except NotImplementedError:
    from psutil import cpu_count
    
from os import getcwd, mkdir, path
import time

from joblib import delayed, Parallel
from keras import layers, models
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

import datahandler as dh


__author__ = 'Gregory Giecold'
__copyright__ = 'Copyright 2017-2022 Gregory Giecold and contributors'
__credit__ = 'Gregory Giecold'
__status__ = 'beta'
__version__ = '0.1.0'


__all__ = ['build_CNN_RNN_hybrid', 'build_dense_baseline', 'build_RNN', 
           'build_RNN_baseline', 'plot_training_history', 
           'simple_baseline']


def simple_baseline(generator, steps, feature_idx):

    try:
        assert isgenerator(generator)
        assert isinstance(steps, int) and steps > 0
        assert isinstance(feature_idx, int)
    except AssertionError:
        raise

    with Parallel(n_jobs=cpu_count(), backend='multiprocessing') as parallel:
        mae_scores = parallel(delayed(simple_baseline_helper)(
            samples, targets, feature_idx) for _, (samples, targets) in zip(range(steps), generator))

    return np.mean(mae_scores)


def simple_baseline_helper(samples, targets, feature_idx):

    predictions = samples[:, -1, feature_idx]    
    mae = np.mean(np.abs(predictions - targets))
    return mae
                              

def build_dense_baseline(input_shape):

    try:
        assert isinstance(input_shape, tuple)
    except AssertionError:
        raise
    
    model = models.Sequential()
    
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.summary()
        
    model.compile(optimizer='rmsprop', loss='mae')

    return model


def build_RNN_baseline(input_shape):
    
    try:
        assert isinstance(input_shape, tuple)
    except AssertionError:
        raise
    
    model = models.Sequential()

    model.add(layers.GRU(32, input_shape=(None, input_shape[-1])))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(lr=0.001), loss='mae')

    return model


def build_RNN(input_shape, dropout=0.5, recurrent_dropout=0.5,
              stacked=False, bidirectional=False):

    try:
        assert isinstance(input_shape, tuple)
        assert isinstance(dropout, (int, float)) and 0 <= dropout < 1
        assert isinstance(recurrent_dropout, (int, float)) and 0 <= recurrent_dropout < 1
        assert isinstance(stacked, (bool, int))
        assert isinstance(bidirectional, (bool, int))
    except AssertionError:
        raise

    model = models.Sequential()

    if bidirectional:
        model.add(layers.Bidirectional(
            layers.GRU(32, input_shape=(None, input_shape[-1]))))
        model.add(layers.Dense(1))
    else:
        model.add(layers.GRU(32, dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            input_shape=(None, input_shape[-1]),
            return_sequences=True if stacked else False))
        if stacked:
            model.add(layers.GRU(64, activation='relu', dropout=dropout,
                recurrent_dropout=recurrent_dropout))
        model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(lr=0.001), loss='mae')
    
    return model


def build_CNN_RNN_hybrid(input_shape, kernel_size=5, pool_size=3,
                         dropout=0.5, recurrent_dropout=0.5):
    
    try:
        assert isinstance(input_shape, tuple)
        assert isinstance(kernel_size, int) and kernel_size > 0
        assert isinstance(pool_size, int) and pool_size > 0
        assert isinstance(dropout, (int, float)) and 0 <= dropout < 1
        assert isinstance(recurrent_dropout, (int, float)) and 0 <= recurrent_dropout < 1
    except AssertionError:
        raise

    model = models.Sequential()

    model.add(layers.Conv1D(32, kernel_size, activation='relu',
        input_shape=(None, input_shape[-1])))
    model.add(layers.MaxPooling1D(pool_size))
    model.add(layers.Conv1D(32, kernel_size, activation='relu'))
    model.add(layers.GRU(32, dropout=dropout,
        recurrent_dropout=recurrent_dropout))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(lr=1e-3), loss='mae')

    return model


def plot_training_history(history, fname):

    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b+')
    plt.plot(epochs, validation_loss, 'bo')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')

    with open(fname, 'w') as fh:
        plt.savefig(fname)
    
    plt.show()
    

def main():

    try:
        odir = path.join(path.dirname(getcwd()),
            'output', 'weather_forecasting')
        mkdir(odir)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
    
    url = "https://s3.amazonaws.com/keras-datasets/" +\
          "jena_climate_2009_2016.csv.zip"
    fname = dh.store_data(url)
    df = dh.to_dataframe(fname, sep=',')

    target_feature = 'T (degC)'
    feature_idx = df.columns.tolist().index(target_feature)
    
    train_days = 1000
    validation_days = 1000

    lookahead = 1 # unit: day
    lookback = 5 # unit: day
    batch_size = 128
    
    df = dh.normalize(df, train_days)

    train_gen = dh.data_generator(df, end_idx=train_days, lookahead=lookahead,
                                  lookback=lookback, batch_size=batch_size)
    validation_gen = dh.data_generator(df, begin_idx=train_days,
                                       end_idx=train_days + validation_days,
                                       lookahead=lookahead, lookback=lookback,
                                       batch_size=batch_size)
    test_gen = dh.data_generator(df, begin_idx=train_days + validation_days,
                                 lookahead=lookahead, lookback=lookback,
                                 batch_size=batch_size, timeout=True)

    train_steps = 10 * train_days // batch_size
    validation_steps = 10 * validation_days // batch_size
    test_steps = 10 * (df.index.levshape[0] - train_days - validation_days) \
                 // batch_size

    mae_score = simple_baseline(validation_gen, validation_steps, feature_idx)
    print("\n\nMean Absolute Error (MAE) of a simple baseline consisting "
          "in predicting the last temperature reading of "
          "each training sample: {mae_score}\n\n".format(**locals()))

    samples, _ = next(train_gen)
    input_shape = samples[0, :, :].shape
    
    experiments = (build_dense_baseline, build_RNN_baseline, 
                   build_CNN_RNN_hybrid, build_RNN)
    hyperparameters = (None, None, (0.1, 0.5, 5, 3), (0.2, 0.2, False, False),
        (0.1, 0.5, True, False), (0, 0, False, True))
    
    for experiment, hyperparams in zip_longest(experiments, hyperparameters, fillvalue=build_RNN):
        if hyperparams is None:
            model = experiment(input_shape)
        elif experiment.__name__ == 'build_CNN_RNN_hybrid':
            dropout, recurrent_dropout, kernel_size, pool_size = hyperparams
            model = experiment(input_shape, kernel_size, pool_size,
                dropout, recurrent_dropout)
        else:
            dropout, recurrent_dropout, stacked = hyperparams
            model = experiment(input_shape, dropout,
                recurrent_dropout, stacked)
        
        epochs = 20
        if experiment.__name__ == 'build_RNN':
            if hyperparams is not None and hyperparams[0] > 0:
                epochs *= 2
            else:
                spec = getargspec(experiment)
                defaults = dict(zip(spec.args[::-1], (spec.defaults or ())[::-1]))
                defaults.update(spec.keywords or {})

                if defaults['dropout'] > 0:
                    epochs *= 2
            
        history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                                      epochs=epochs, validation_data=validation_gen,
                                      validation_steps=validation_steps)
        
        fname = path.join(odir, '_'.join([
            experiment.__name__.lstrip('build_'), '' if hyperparams is None \
            else '_'.join(map(lambda x: str(x), hyperparams)),
            'losses.png']))
        plot_training_history(history, fname)

    
if __name__ == '__main__':

    main()
    
