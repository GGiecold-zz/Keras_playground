#!/usr/bin/env python


r"""Train a convolutional neural network with dropout
    and data-augmentation on the Kaggle dogs-versus-cats dataset.
    Since the number of training samples is fairly small, also
    illustrated is the use of various pre-trained convnets (such as
    Inception or VGG16) for feature-extraction or fine-tuning;
    a densely-connected classifier is then trained on top of such
    convolutional base.
"""


from __future__ import print_function

from builtins import range, zip
import errno
from itertools import product
from os import getcwd, makedirs, mkdir, path, rename
import shutil
from time import sleep
import zipfile

from keras import layers, models
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'Gregory Giecold'
__copyright__ = 'Copyright 2017-2022 Gregory Giecold and contributors'
__credit__ = 'Gregory Giecold'
__status__ = 'beta'
__version__ = '0.1.0'


__all__ = ['build_model', 'create_validation_and_test_sets',
           'data_preprocessing', 'ema',
           'plot_loss_and_accuracy']


def build_model(height=150, width=150, pretrained=None, 
                fine_tuning=False, dropout=True):

    assert isinstance(fine_tuning, bool)
    assert isinstance(dropout, bool)
    
    model = models.Sequential()

    if pretrained is not None:
        assert isinstance(pretrained, str)
        assert pretrained in ('InceptionV3', 'ResNet50', 'VGG16',
                              'VGG19', 'Xception')

        module = __import__('keras.applications')
        convolutional_base = getattr(getattr(module, 'applications'), pretrained)(
            include_top=False, weights='imagenet',
            input_shape=(height, width, 3))
        convolutional_base.summary()
        # For simple feature-extraction, freeze the convolutional base, 
        # lest the representations previously learned by it on Imagenet end up 
        # being modified by backpropagation of our classifier's weights through 
        # the whole, stacked network. For fine-tuning, after training a
        # classifier, unfreeze the last 2 or 3 layers of the base network
        # then jointly train those layers and the classifier's weights.
        if fine_tuning is True:
            convolutional_base.trainable = True
            
            trainable_flag = False
            for layer in convolutional_base.layers:
                if layer.name == 'block5_conv1':
                    trainable_flag = True
                layer.trainable = True if trainable_flag else False
        else:
            convolutional_base.trainable = False
        
        model.add(convolutional_base)
        
        model.add(layers.Flatten())

        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
            input_shape=(height, width, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
    
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        if dropout:
            model.add(layers.Dropout(0.5))

        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


def create_validation_and_test_sets(base_path='../data/catsVsDogs.zip'):
    
    assert path.isfile(base_path)
    
    base_dir, _ = path.splitext(base_path) 
    if not path.isdir(base_dir):
        try:
            mkdir(base_dir)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
    
        with zipfile.ZipFile(base_path, 'r') as zh:
            zh.extractall(path.dirname(base_path))
            rename(path.join(path.dirname(base_path), zh.namelist()[0]),
                   base_dir)

    small_dir = '_'.join([base_dir, 'small'])
    subdirs = ('train', 'validation', 'test')
    chunks = ((0, 1000), (1000, 1500), (1500, 2000))
    
    for ((subdir, chunk), category) in product(
            zip(subdirs, chunks), ('cats', 'dogs')):
        current_dir = path.join(small_dir, subdir, category)
        
        try:
            makedirs(current_dir)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

        fnames = (category.rstrip('s') + '.{}.jpg'.format(i)
                  for i in range(*chunk))  # generator
        for fname in fnames:
            shutil.copyfile(path.join(base_dir, fname),
                            path.join(current_dir, fname))

            
def ema(data, l=0.8):

    smoothed = list()

    for current in data:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append((1 - l) * current + l * previous)
        else:
            smoothed.append(current)
            
    return smoothed


def plot_loss_and_accuracy(history):

    dico = history.history
    
    training_losses = dico['loss']
    validation_losses = dico['val_loss']

    epochs = range(1, len(training_losses) + 1)

    plt.plot(epochs, ema(training_losses), 'b+')
    plt.plot(epochs, ema(validation_losses), 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training (blue) and validation (red) loss')
    plt.figure()

    training_accuracies = dico['acc']
    validation_accuracies = dico['val_acc']

    plt.plot(epochs, ema(training_accuracies), 'bo')
    plt.plot(epochs, ema(validation_accuracies), 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training (blue) and validation (red) accuracy')
    plt.show()
    sleep(2)
    plt.close()

    
def data_preprocessing(base_path='../data/catsVsDogs.zip',
                       height=150, width=150,
                       batch_size=20, class_mode='binary',
                       data_augmentation=True):

    assert isinstance(data_augmentation, bool)
    assert path.isfile(base_path)
    
    base_dir, _ = path.splitext(base_path)
    test_dir = path.join('_'.join([base_dir, 'small']), 'test')
    train_dir = path.join('_'.join([base_dir, 'small']), 'train')
    validation_dir = path.join('_'.join([base_dir, 'small']), 'validation')

    if data_augmentation:
        train_generator = ImageDataGenerator(rescale=1/255.0,
            height_shift_range=0.2, width_shift_range=0.2,
            rotation_range=40, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode='nearest').flow_from_directory(
                train_dir, target_size=(height, width),
                batch_size=batch_size, class_mode=class_mode)
    else:
        train_generator = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
            train_dir, target_size=(height, width),
            batch_size=batch_size, class_mode=class_mode)

    validation_generator = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
        validation_dir, target_size=(height, width),
        batch_size=batch_size, class_mode=class_mode)

    test_generator = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
        test_dir, target_size=(height, width),
        batch_size=batch_size, class_mode=class_mode)
    
    return train_generator, validation_generator, test_generator

            
def main():

    training_samples = 2000
    validation_samples = test_samples = 1000

    create_validation_and_test_sets()

    # The first experiment with a pre-trained VGG16
    # uses that convnet for feature-extraction; the second
    # such experiment uses that model for fine-tuning, i.e.
    # unfreezing the top few layers and training them jointly
    # with a classifier. Note that it is highly advisable 
    # to use an already-trained classifier; otherwise the 
    # randomly-initialized weights might turn into too large
    # a back-propagated error signal through the convolutional
    # base, thereby destroying the representations
    # previously learned.
    hyperparameters_generator = zip(
        (False, True, True, True), (None, None, 'VGG16', 'VGG16'), 
        (False, False, False, True), (False, True, False, False), 
        (20, 32, 20, 20)
    )
    
    for i, hyperparameters in enumerate(hyperparameters_generator, 1):
        data_augmentation, pretrained, fine_tuning, \
            dropout, batch_size = hyperparameters
            
        train_generator, validation_generator, test_generator = data_preprocessing(
            data_augmentation=data_augmentation,
            batch_size=batch_size)
    
        model = build_model(pretrained=pretrained,
                            fine_tuning=fine_tuning,
                            dropout=dropout)
        model.summary()
        model.compile(optimizer=RMSprop(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        if (pretrained is None) and (data_augmentation is False):
            history = model.fit_generator(
                train_generator, epochs=30,
                steps_per_epoch=training_samples/batch_size,
                validation_data=validation_generator,
                validation_steps=validation_samples/batch_size)
        elif (pretrained is not None) and (data_augmentation is True):
            history = model.fit_generator(
                train_generator, epochs=100,
                steps_per_epoch=training_samples/batch_size,
                validation_data=validation_generator,
                validation_steps=validation_samples/batch_size)
        else:
            history = model.fit_generator(
                train_generator, epochs=30,
                samples_per_epoch=training_samples,
                validation_data=validation_generator,
                nb_val_samples=validation_samples)

        plot_loss_and_accuracy(history)

        model.save(path.join(getcwd(),
            'catsVsDogs_small_convnet_experiment_{}.h5'.format(i)))

        test_loss, test_accuracy = model.evaluate(
            test_generator, test_samples=test_samples)

        print('\n\nExperiment {i}: test accuracy: {test_accuracy}\n\n'.format(
            **locals()))

        
if __name__ == "__main__":
    
    main()
    
