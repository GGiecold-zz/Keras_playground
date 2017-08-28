#!/usr/bin/env python


r"""Training a sentiment analysis binary classifier on the IMDB dataset,
    with or without pretrained GloVe word embeddings.

    Downloading and extracting the various models and datasets involved is done
    in parallel, along with running various make files and scripts.
"""


from __future__ import print_function

from builtins import zip
from collections import MutableSequence, Sequence
import fnmatch
import functools
from inspect import isgenerator
from multiprocessing import Pool
import os
from os import getcwd, listdir, mkdir, path

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

import shlex
import six
import subprocess
import sys
import tarfile

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock
    
import zipfile

from keras import layers, models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import wget


__author__ = 'Gregory Giecold'
__copyright__ = 'Copyright 2017-2022 Gregory Giecold and contributors'
__credit__ = 'Gregory Giecold'
__status__ = 'beta'
__version__ = '0.1.0'


__all__ = ['build_model', 'download_extract',
           'get_imdb_data', 'tokenize_data',
           'track_train']


def track_train(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        train_flag = kwargs.pop('train_flag', None)
        if train_flag is not None:
            wrapper.has_been_trained = train_flag
        return func(*args, **kwargs)
    
    wrapper.has_been_trained = False
    return wrapper
    

def download_extract(url, odir):

    assert isinstance(url, six.string_types)
    
    if url.endswith('.gz'):
        fname = wget.download(url, out=path.dirname(odir), bar=None)
        
        with tarfile.open(fname, 'r') as th:
            th.extractall(path.dirname(odir))

        subprocess.check_call(['rm', '{}'.format(path.split(url)[-1])],
                              stdout=DEVNULL, stderr=subprocess.PIPE,
                              cwd=path.dirname(odir))
    elif url.endswith('GloVe-1.2.zip'):
        fname = wget.download(url, out=path.dirname(odir),
                              bar=wget.bar_thermometer)
        
        with zipfile.ZipFile(fname, 'r', zipfile.ZIP_DEFLATED) as zh:
            for file in zh.filelist:
                name = file.filename
                permissions = 0777 
                file.external_attr = permissions
                ofile = path.join(path.dirname(odir), name)
                
                if name.endswith('/'):
                    mkdir(ofile, permissions)
                else:
                    fh = os.open(ofile, os.O_CREAT | os.O_WRONLY, permissions)
                    os.write(fh, zh.read(name))
                    os.close(fh)

        commands = ('rm {}'.format(path.split(url)[-1]), 'make', './demo.sh')
        directories = (path.dirname(odir), odir, odir)
        shell_flags = (False, False, True)
        
        for cmd, cdir, flag in zip(commands, directories, shell_flags):
            subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             cwd=cdir, shell=flag)
    elif url.endswith('.zip'):
        fname = wget.download(url, out=path.dirname(odir))
        
        with zipfile.ZipFile(fname, 'r') as zh:
            zh.extractall(odir)
            
        subprocess.check_call(['rm', '{}'.format(path.split(url)[-1])],
                              stdout=DEVNULL, stderr=subprocess.PIPE,
                              cwd=path.dirname(odir))

    
def download_extract_unpack(args):

    return download_extract(*args)
    
        
def get_imdb_data(odir, train_flag=True):

    assert path.isdir(odir)
    assert isinstance(train_flag, bool)

    labels, texts = list(), list()
    for category in ('neg', 'pos'):
        subdir = path.join(odir, 'train' if train_flag else 'test', category)
        for fname in fnmatch.filter(listdir(subdir), '*.txt'):
            labels.append(0 if category == 'neg' else 1)
            
            with open(path.join(subdir, fname), 'r') as fh:
                texts.append(fh.read())
    
    return labels, texts


def tokenize_data(tokenizer, odir, num_words, num_training_samples=20000,
    num_validation_samples=10000, max_words_per_text=100):
    
    @track_train
    def helper(train_flag=True):
        
        labels, texts = get_imdb_data(odir, train_flag)
        
        labels = np.asarray(labels)
        
        try:        
            if isinstance(texts, (MutableSequence, Sequence)):
                texts = list(texts)
            else:
                assert isgenerator(texts)
        except Exception:
            raise
        
        if train_flag:
            tokenizer.fit_on_texts(texts)
    
        data = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(data, maxlen=max_words_per_text)

        return labels, data

    labels, data = helper()
    
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)

    data = data[idx]
    labels = labels[idx]

    x_train = data[:num_training_samples]
    y_train = labels[:num_training_samples]
    x_val = data[
        num_training_samples:num_training_samples + num_validation_samples
    ]
    y_val = labels[
        num_training_samples:num_training_samples + num_validation_samples
    ]
    
    y_test, x_test = helper(False)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(pretrained_embedding=True, odir=None, tokenizer=None,
                num_words=10000, embedding_dimension=100,
                max_words_per_text=100):
    
    if pretrained_embedding:
        assert embedding_dimension in (50, 100, 200, 300)
        
    model = models.Sequential()

    model.add(layers.Embedding(
        num_words, embedding_dimension,
        input_length=max_words_per_text)
    )
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    if pretrained_embedding:
        assert (odir is not None) and path.isdir(odir)
        assert (tokenizer is not None) and isinstance(tokenizer, Tokenizer)
        
        embedding_dict = dict()
        fname = path.join(
            odir, 'glove.6B.{embedding_dimension}d.txt'.format(**locals())
        )

        with open(fname, 'r') as fh:
            for line in fh:
                tmp = line.split()
                k, v = tmp[0], tmp[1:]
                
                embedding_dict[k] = np.asarray(v, dtype='float32')

        embedding_matrix = np.zeros((num_words, embedding_dimension))
        
        for k, v in tokenizer.word_index.iteritems():
            word_embedding = embedding_dict.get(k)
            if v < num_words and word_embedding is not None:
                embedding_matrix[v] = word_embedding
        
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False
        
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

    
def main():

    odir = path.join(path.dirname(getcwd()), 'data')

    imdb_dir = path.join(odir, 'aclImdb')
    glove_code_dir = path.join(odir, 'GloVe-1.2')
    pretrained_glove_embedding_dir = path.join(odir, 'glove.6B')
    
    dirs, urls = list(), list()
    if not path.isdir(glove_code_dir):
        dirs.append(glove_code_dir)
        urls.append('https://nlp.stanford.edu/software/GloVe-1.2.zip')
    if not path.isdir(pretrained_glove_embedding_dir):
        dirs.append(pretrained_glove_embedding_dir)
        urls.append('http://nlp.stanford.edu/data/glove.6B.zip')
    if not path.isdir(imdb_dir):
        dirs.append(imdb_dir)
        urls.append(
            'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        )

    pool = Pool()
    if sys.version_info.major == 3:
        pool.starmap(download_extract, zip(urls, dirs))
    else:
        pool.map(download_extract_unpack, zip(urls, dirs))

    num_words = 10000
    tokenizer = Tokenizer(num_words=num_words)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = tokenize_data(
        tokenizer, imdb_dir, num_words)

    for pretrained in (True, False):
        model = build_model(pretrained, pretrained_glove_embedding_dir,
                            tokenizer)
        
        history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_data=(x_val, y_val), verbose=0)
        scores = model.evaluate(x_test, y_test, verbose=0)
        
        print("\nTest results for the model "
              "{} pretrained GloVe word embeddings: ".format(
              'with' if pretrained else 'without'))
        print("loss={scores[0]}, accuracy={scores[1]}\n".format(**locals()))
        # The model with pretrained embedding vectors will display a lower
        # test accuracy, due to having overfitted the training samples.
        
        model.save_weights('{}glove_model.hy'.format(
            'pretrained_' if pretrained else ''))


if __name__ == '__main__':

    main()
    
