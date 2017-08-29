#!/usr/bin/env python


r"""Gathering and processing the data required for predicting 
    daily temperatures from temperature, atmospheric pressure
    and hygrometry time series.

    Instead of gathering all daily time series into a single large
    array, careful consideration is given to the possibility of
    duplicated rows, missing measurements, etc.
    This is accomplished by massaging the data into a multi-indexed
    dataframe (in the case at hand a two-level index accessed by
    specifying a day and a timestamp).
    
    Indeed, it turns out that for the so-called Jena dataset there are 
    287 timestamps on '2010-01-07', instead of the expected 144 
    (on most days measurements are recorded every 10 minutes).
    
    As another illustration, '2016-10-25' comes with only 64 timestamps; 
    for some unaccounted reason, measurements range only from 12:00:00 A.M.
    to 10:30:00 A.M. on that day.

    A related issue addressed in the code herewith is that of
    calendar gaps. This would be overlooked by simply aggregating
    all time series in an array and would affect the purpose
    of making forecasts one or more days ahead.
"""


from __future__ import print_function

from builtins import enumerate, map, range, zip
import datetime
from itertools import permutations, product
import operator
from os import devnull, getcwd, path
import random
import shlex
import signal
import six
import subprocess
import sys
import tarfile

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import warnings
import zipfile

import numpy as np
import pandas as pd
from sortedcontainers import SortedSet
import wget


__author__ = 'Gregory Giecold'
__copyright__ = 'Copyright 2017-2022 Gregory Giecold and contributors'
__credit__ = 'Gregory Giecold'
__status__ = 'beta'
__version__ = '0.1.0'


__all__ = ['data_generator', 'date_parser', 'fill_time', 'normalize',
           'resample', 'store_data', 'timeout_handler',
           'to_batchsize', 'to_dataframe', 'to_timedelta']


class TimeoutException(Exception):

    pass


def timeout_handler(signalnum, frame):

    raise TimeoutException


def to_timedelta(s, fmt='%H:%M:%S'):

    try:
        assert isinstance(s, six.string_types)
    except AssertionError:
        raise
    
    return datetime.datetime.strptime(s, fmt) -\
        datetime.datetime.strptime('00:00:00', fmt)


def store_data(url, odir=path.join(path.dirname(getcwd()), 'data')):

    try:
        assert path.isdir(odir)
    except AssertionError:
        raise ValueError('Incorrect directory provided\n')
        sys.exit(1)

    fname = path.split(url)[-1]
    
    try:
        if fname.endswith('.gz'):
            with tarfile.open(wget.download(url, out=odir)) as th:
                th.extractall(odir)
        elif fname.endswith('.zip'):
            with zipfile.ZipFile(wget.download(url, out=odir)) as zh:
                zh.extractall(odir)
        else:
            res = urlopen(url)
            chunk_size = 64 * 1024
            with open(path.join(odir, fname), 'wb') as fh:
                while True:
                    chunk = res.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
    except tarfile.ReadError, zipfile.BadZipfile:
        sys.exit(1)
    except tarfile.ExtractError, zipfile.error:
        sys.exit(1)

    if fname.endswith(('.gz', '.zip')):
        cmd = 'rm {}'.format(fname)
        DEVNULL = open(devnull, 'wb')
        subprocess.Popen(shlex.split(cmd), stdout=DEVNULL,
                         stderr=subprocess.PIPE, cwd=odir)

    return path.join(odir, fname.rstrip('.gz').rstrip('.tar').rstrip('.zip'))


def date_parser(s):

    try:
        if isinstance(s, pd.Timestamp):
            s = s.date().strftime('%d.%m.%Y')
        elif isinstance(s, datetime.datetime):
            s = s.strftime('%d.%m.%Y')
        elif isinstance(s, datetime.date):
            s = '.'.join([s.day, s.month, s.year])
            
        assert isinstance(s, six.string_types)
    except AssertionError:
        raise
        
    separators = ('.', '-', '/', ':', '')
    
    for sep in separators:
        formats = map(
            lambda x: sep.join(x),
            permutations(('%d', '%m', '%Y'), 3)
        )
        for fmt in formats:
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                pass

    raise ValueError('Invalid date format\n')

    
def to_dataframe(fname, sep=None):

    try:
        assert path.isfile(fname)
    except AssertionError:
        raise ValueError('No such file\n')
        sys.exit(1)

    try:
        assert (sep is None) or isinstance(sep, six.string_types)
    except AssertionError:
        raise ValueError('Invalid separator provided\n')
        sys.exit(1)

    columns = pd.read_table(fname, sep, header='infer', nrows=0)
    columns = columns.columns.tolist()
    for elt in ('Date Time', 'Date time', 'Datetime'):
        try:
            columns.remove(elt)
            columns.remove(elt.lower())
        except ValueError:
            pass
        
    if sep is None:
        sep = r'\s+'
    else:
        sep = r'[\s' + sep + ']'
    
    df = pd.read_table(
        fname, sep, header=None, skiprows=1, engine='python',
        names = ['Date', 'Time'] + columns,
        converters={'Time': lambda x: datetime.time(*map(int, x.split(':'))),
                    'Date': lambda x: date_parser(x)},
        index_col=('Date', 'Time'),
        infer_datetime_format=True,
        na_filter=True, skip_blank_lines=True,
        dayfirst=False, compression=None,
        comment='#', error_bad_lines=True
    )
    df.drop_duplicates(inplace=True)
    df.index = pd.MultiIndex.from_tuples(
        list(map(lambda tpl: (pd.to_datetime(tpl[0]), tpl[1]), df.index)),
        names=['Date', 'Time']
    )
    df.sort_index(inplace=True)
    
    print(df.head(), '\n\n')
    sys.stdout.flush()

    days = df.index.levels[0]
    warnings.simplefilter('always', UserWarning)
    for current_day, next_day in zip(days[:-1], days[1:]):
        try:
            delta = next_day - current_day
            assert delta.days == 1
        except AssertionError:
            fmt = '%Y-%m-%d'
            current_day = date_parser(current_day).strftime(fmt)
            next_day = date_parser(next_day).strftime(fmt)
            
            msg = "\n{} and {} ".format(current_day, next_day)
            msg += "are consecutive in the first level of the multi-index "
            msg += "but exhibit a calendar gap; this might have implications "
            msg += "for your forecasts and training your models.\n\n"
            warnings.warn(msg, UserWarning)
            
            continue
    
    return df


def normalize(df, training_days, with_mean=True, with_std=True):

    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(training_days, int) and training_days > 0
    except AssertionError:
        raise

    training_days = min(training_days, df.index.levshape[0])

    days = df.index.levels[0]
    tmp = df.loc[(days[:training_days], slice(None)), :]
    
    mean = tmp.mean(axis=0) if with_mean else 0.0
    std = tmp.std(axis=0) if with_std else 1.0

    df -= mean
    df /= std

    return df


def data_generator(df, target_feature='T (degC)',
                   begin_idx=None, end_idx=None,
                   lookahead=1, lookback=5, sampling_rate='1h',
                   batch_size=128, shuffle=True, timeout=False):
    """
       Parameters
       ----------
       df: type pandas.DataFrame
           A multi-index structure is assumed, i.e. a hierarchy of
           rows with the first level corresponding to a date 
           and the second level to a timestamp.

       target_feature: type six.string_types (default: 'T (degC)')
           Refers to a column/feature name in 'df' for which
           lookahead targets should be extracted.
 
       begin_idx: type int (default: None)
           Denotes an index in the 'df' dataframe corresponding to the earliest
           date from to begin drawing samples.

       end_idx: type int (default: None)
           Denotes and index in the 'df' dataframe corresponding to the latest
           day from which to draw.

       lookahead: type int (default: 1)
           How many day(s) in the future the forecast target should be.

       lookback: type int (default: 5)
           How many days back to gather input data (one sample).

       sampling_rate: type six.string_types (default: '1h')
           Over which period to average data points.

       batch_size: type int (default: 128)
           Number of samples per batch.

       shuffle: type bool (default: True)
           Whether to shuffle the days or fetch them in order.

       timeout: type bool (default: False)
           Used to specify how long to wait for a batch of samples
           to be processed. This flag has to be set to False
           if you were to fork to multiple processes; indeed,
           only the main thread can set or invoke signal handlers,
           and in the code under present consideration an alarm signal 
           handler is underlying our implementation of the 
           timeout notification.

       Returns
       -------
       tuple of ('samples', 'targets'), where samples correspond to one
       batch of input data ('batch_size' of them, each sample corresponding
       to 'lookback' full days) and where 'targets' denotes an array of
       target temperatures.
    """
    
    begin_idx = 0 if begin_idx is None else begin_idx
    if end_idx is None:
        end_idx = df.index.levshape[0] - lookahead

    try:
        assert isinstance(target_feature, six.string_types)
        
        assert isinstance(begin_idx, int)
        assert isinstance(end_idx, int)
        assert begin_idx < df.index.levshape[0]
        assert begin_idx <= end_idx
    except AssertionError:
        raise

    days = df.index.levels[0].to_datetime()
    idx = begin_idx + lookback
    
    while True:
        if shuffle:
            day_indices = np.random.randint(idx, end_idx, batch_size)
        else:
            if idx + batch_size >= end_idx:
                idx = begin_idx + lookback
            day_indices = np.arange(idx, min(idx + batch_size, end_idx))
            idx += random_indices.size
            
        samples, targets = list(), list()

        for day_idx in day_indices:
            past_day = days[day_idx - lookback + 1]
            current_day = days[day_idx]
            next_day = days[day_idx + lookahead]

            delta = next_day - current_day
            if delta.days != lookahead:
                pass

            sample = resample(df, sampling_rate, current_day, past_day)
            tmp = resample(df, sampling_rate, next_day)

            if timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
            while True:
                if timeout:
                    signal.alarm(batch_size // 5)
                    
                try:
                    time_idx, timestamp = random.choice(list(enumerate(tmp.index)))
                    timestamp = timestamp.time().isoformat()
                    try:
                        key = current_day.date().isoformat() + ' ' + timestamp
                        sample.loc[key, :]
                        break
                    except KeyError:
                        pass
                except TimeoutException:
                    continue
                else:
                    if timeout:
                        signal.alarm(0)
                
            past_datetime = ' '.join([past_day.date().isoformat(),
                                      timestamp])
            current_datetime = ' '.join([current_day.date().isoformat(),
                                         timestamp])

            sample = fill_time(sample, sampling_rate)
            sample = sample.loc[slice(past_datetime, current_datetime), :].values
            target = tmp.loc[tmp.index[time_idx], target_feature]

            samples.append(sample)
            targets.append(target)

        samples = np.asarray(samples)
        targets = np.asarray(targets)
        samples, targets = to_batchsize(samples, targets, batch_size)

        yield samples, targets


def resample(df, sampling_rate, current_day, past_day=None):

    if past_day is not None:
        sample = df.loc[(slice(past_day, current_day), slice(None)), :]
        newindex = map(lambda tpl: operator.add(*tpl),
            zip(sample.index.get_level_values('Date'),
                map(to_timedelta, sample.index.get_level_values('Time'))
            )
        )
        sample.index = list(newindex)
        sample = sample.resample(sampling_rate).mean()
    else:
         sample = df.xs(current_day, level='Date')
         sample.index = pd.to_datetime(sample.index)
         sample = sample.resample(sampling_rate).mean()

    return sample


def fill_time(df, sampling_rate='1h'):
    r"""This procedure ensures that if a batch of 'lookback' days
        were to feature a day with less measurements than most
        (say, only one afternoon of measurements, instead of the usual 
         full range - from 00:00:00 till 23:50:00), the batch size
        will still share a common value across all batches. This is
        simply achieved by filling in and interpolating missing 
        time measurements.

       The possibility of a calendar gap within the measurements
       held in dataframe 'df' is properly accounted: we do not fill 
       the intervening day(s) and timestamps.
    """
        
    try:
        assert isinstance(df.index, pd.DatetimeIndex)
    except AssertionError:
        raise TypeError("Provide a dataframe indexed by a DatetimeIndex; "
                        "multi-indexed dataframes are not supported in "
                        "the present version.\n")

    try:
        assert sampling_rate in ('1h',)
    except AssertionError:
        raise ValueError('No other sampling rate supported for now.\n')
    # This could be easily addressed if the need to experiment
    # with different resampling formats were to arise.

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)
    
    earliest_day = df.index[0].date().isoformat()
    latest_day = df.index[-1].date().isoformat()

    gap_pairs = list()
    days = SortedSet(map(lambda x: x.date(), df.index))
    for current_day, next_day in zip(days[:-1], days[1:]):
        delta = next_day - current_day
        if delta.days > 1:
            gap_pairs.append((current_day, next_day))
    gap_pairs.reverse()

    # Would need to expand this if more options were introduced
    # for the choice of resampling schedule.
    if sampling_rate == '1h':
        earliest_time = '00:00:00'
        latest_time = '23:00:00'

    if not gap_pairs:
        idx = pd.date_range(' '.join([earliest_day, earliest_time]),
            ' '.join([latest_day, latest_time]), freq=sampling_rate)
    else:
        previous_beg_gap, previous_end_gap = gap_pairs.pop()
        idx = pd.date_range(' '.join([earliest_day, earliest_time]),
            ' '.join([previous_beg_gap, latest_time]), freq=sampling_rate)
        
        while gap_pairs:
            beg_gap, end_gap = gap_pairs.pop()
            idx = idx.append(pd.date_range(' '.join([previous_end_gap, earliest_time]),
                                           ' '.join([beg_gap, latest_time]),
                                           freq=sampling_rate))
            previous_end_gap = end_gap

        idx = idx.append(pd.date_range(' '.join([previous_end_gap, earliest_time]),
                                       ' '.join([latest_day, latest_time]),
                                       freq=sampling_rate))
                
    df = df.reindex(idx, fill_value=np.nan, copy=False)
    
    df.ffill(axis=0, inplace=True)
    df.bfill(axis=0, inplace=True)
    
    return df

         
def to_batchsize(samples, targets, batch_size):
    r"""This procedure expands samples and targets to the required
        size, i.e. 'batch_size'. 
        Indeed, an array 'samples' could arise whose first dimension
        has less than the required number of samples because
        our data_generator does not create a ('sample', 'target') pair
        when there exists a calendar gap of more than one 'lookahead'
        day(s) between the day on which 'target' was measured and the
        latest day recorded in 'sample'.
    """

    try:
        assert isinstance(batch_size, int) and batch_size > 0
        
        assert isinstance(samples, np.ndarray)
        assert isinstance(targets, np.ndarray)
        
        assert targets.ndim == 1
        
        targets = targets.reshape(targets.size)
        
        assert samples.shape[0] == targets.size
    except AssertionError:
        raise
    
    if targets.size < batch_size:
        repeats = [1] * targets.size

        i = -1
        diff = batch_size - targets.size 
        while diff:
            repeats[i % targets.size] += 1
            i -= 1
            diff -= 1
                
        samples = np.repeat(samples, repeats, axis=0)
        targets = np.repeat(targets, repeats)
    elif targets.size > batch_size:
        samples = samples[:batch_size]
        targets = targets[:batch_size]

    reshuffle_idx = np.random.permutation(batch_size)
    samples = samples[reshuffle_idx]
    targets = targets[reshuffle_idx]

    return samples, targets
    
