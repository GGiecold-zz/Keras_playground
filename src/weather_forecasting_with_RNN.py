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
from inspect import isgenerator

try:
    from multiprocessing import cpu_count
except NotImplementedError:
    from psutil import cpu_count
    
from os import getcwd, mkdir, path
import time

from joblib import delayed, Parallel
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

import datahandler as dh


__author__ = 'Gregory Giecold'
__copyright__ = 'Copyright 2017-2022 Gregory Giecold and contributors'
__credit__ = 'Gregory Giecold'
__status__ = 'beta'
__version__ = '0.1.0'


__all__ = ['build_dense_model', 'plot_training_history', 'simple_baseline']


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
                              

def build_dense_model(input_shape):

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

    plt.show()
    plt.savefig(fname)
    
    time.sleep(2)
    plt.close()
    

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
    
    training_days = 1000
    validation_days = 1000

    lookahead = 1 # unit: day
    lookback = 5 # unit: day
    batch_size = 128
    
    df = dh.normalize(df, training_days)

    train_gen = dh.data_generator(df, end_idx=training_days,
                                  lookahead=lookahead, lookback=lookback,
                                  batch_size=batch_size)
    validation_gen = dh.data_generator(df, begin_idx=training_days,
                                       end_idx=training_days + validation_days,
                                       lookahead=lookahead, lookback=lookback,
                                       batch_size=batch_size)
    test_gen = dh.data_generator(df, begin_idx=training_days + validation_days,
                                 lookahead=lookahead, lookback=lookback,
                                 batch_size=batch_size, timeout=True)

    validation_steps = validation_days // batch_size
    test_steps = (df.index.levshape[0] - training_days - validation_days) \
                 // batch_size

    mae_score = simple_baseline(validation_gen, validation_steps, feature_idx)
    print("\n\nMean Absolute Error (MAE) of a simple baseline consisting "
          "in predicting the last temperature reading of "
          "each training sample: {mae_score}\n\n".format(**locals()))

    samples, targets = next(train_gen)
    model = build_dense_model(input_shape=samples[0,:,:].shape)
    history = model.fit_generator(
        train_gen, steps_per_epoch=5*training_days // batch_size,
        epochs=20, validation_data=validation_gen,
        validation_steps=5*validation_steps
    )
    fname = path.join(odir, 'densely_connected_net_losses.png')
    plot_training_history(history, fname)

    
if __name__ == '__main__':

    main()
    
