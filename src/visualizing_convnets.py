#!/usr/bin/env python


from __future__ import print_function

from builtins import map, range, zip
import errno
from itertools import product
from os import getcwd, makedirs, path
import random

import cv2
from keras import backend, layers, models
from keras.applications import vgg16
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


__author__ = 'Gregory Giecold'
__copyright__ = 'Copyright 2017-2022 Gregory Giecold and contributors'
__credit__ = 'Gregory Giecold'
__status__ = 'beta'
__version__ = '0.1.0'


__all__ = ['class_activation_preprocess', 'class_activation_heatmap',
           'display_filters', 'display_intermediate_activations',
           'maximally_responsive_pattern', 'to_valid_img', 'trial_1',
           'trial_2', 'trial_3']


def class_activation_heatmap(file_path, architecture='vgg16',
    class_idx=None, odir=getcwd()):
    """This function implements the idea exposed in 
       "Grad-CAM: Visual Explanation from Deep Networks via "
       "Gradient-based Localization', Selvaraju, Cogswell et al., 
       arXiv:1610.02391 [cs.CV]. The gist of it consists in weighting, 
       given an input image, a spatial map of different channel activations 
       by the gradient of a class with respect to those channels. 
       The said channels are assumed to be part of a specific feature map;
       in the present implementation, this feature map is enforced to be the last
       layer of the convolutional base of either of an Inception, ResNet,
       VGG16, VGG19 or Xception architecture (as specified by the eponymous
       parameter).
    """

    # The following is required because we are making use
    # of functions present in Keras for decoding those
    # model's class predictions, as well as functions
    # wrapping some of the image preprocessing steps
    # peculiar to each of those models
    assert architecture in ('inception_v3', 'resnet50','vgg16',
                            'vgg19', 'xception')
    module = getattr(__import__('keras.applications'), 'applications')
    
    if architecture.startswith('vgg'):
        cls = getattr(getattr(module, architecture), architecture.upper())
    elif architecture == 'inception_v3':
        cls = getattr(getattr(module, architecture), 'InceptionV3')
    elif architecture == 'resnet50':
        cls = getattr(getattr(module, architecture), 'ResNet50')
    else:
        cls = getattr(getattr(module, architecture), architecture.capitalize())

    without_top_model = cls(include_top=False, weights='imagenet')
    # without top model corresponds to the convolutional base
    # of the model. It will facilitate access to the last convolution
    # layer via an index instead of having to specify its name. 
    model = cls(weights='imagenet')
    
    model.summary()

    img = class_activation_preprocess(file_path, model.name)

    predictions = model.predict(img)
    
    decoded_predictions = getattr(getattr(module, model.name),
        'decode_predictions')(predictions, top=5)[0]
    print("\n\n\nThe top 5 classes predicted for this image, "
          "and their probabilities are as follows:\n", decoded_predictions)

    # If class_idx defaults to None, then the class with largest predicted
    # probability for the input image will be selected to display
    # the corresponding activation heatmap super-imposed on that image
    if class_idx is None:
        class_idx = np.argmax(predictions[0])
    else:
        assert isinstance(class_idx, int) and 0 <= class_idx < 1000

    class_output = model.output[:, class_idx]
    
    last_convolution_layer = without_top_model.get_layer(index=-2)
    last_convolution_layer = model.get_layer(last_convolution_layer.name)
    
    class_gradients = backend.gradients(class_output,
        last_convolution_layer.output)[0]
    pooled_class_gradients = backend.mean(class_gradients, axis=(0, 1, 2))

    func = backend.function([model.input], [pooled_class_gradients,
        last_convolution_layer.output[0]])

    pooled_class_gradient_values, last_convolution_layer_output = func([img])

    for channel, value in enumerate(pooled_class_gradient_values):
        last_convolution_layer_output[:, :, channel] *= value

    class_activation_heatmap = np.mean(last_convolution_layer_output, axis=-1)
    class_activation_heatmap = np.maximum(class_activation_heatmap, 0)
    class_activation_heatmap /= np.max(class_activation_heatmap)

    plt.matshow(class_activation_heatmap)
    with open(path.join(odir, 'class_{}_heatmap.png'.format(class_idx)), 'w') as fh:
        plt.savefig(fh)

    img = cv2.imread(file_path)

    class_activation_heatmap = cv2.resize(class_activation_heatmap,
        (img.shape[1], img.shape[0]))
    class_activation_heatmap = np.uint8(255 * class_activation_heatmap)
    class_activation_heatmap = cv2.applyColorMap(
        class_activation_heatmap,
        cv2.COLORMAP_JET)

    img = img + 0.4 * class_activation_heatmap
        
    cv2.imwrite(path.join(odir, 'class_{}_superimposed_heatmap.png'.format(
        class_idx)), img)
    
    plt.show()    

    
def class_activation_preprocess(file_path, architecture='vgg16'):
    """The preprocessing steps embodied in the present function
       assume that the model whose class activation heatmap
       we want to display was trained on input images
       of the same format as those fed to the convnet whose
       architecture is specified as one of this function's parameters.
       VGG16 was trained on images of size 224 * 224, with
       some further preprocessing summed up in the function
       'keras.applications.vgg16.preprocess_input'.
    """

    assert path.isfile(file_path)
    
    from keras.preprocessing import image

    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    module = getattr(__import__('keras.applications'), 'applications')
    img = getattr(getattr(module, architecture), 'preprocess_input')(img)

    return img


def maximally_responsive_pattern(model, layer_name, filter_index=0,
    img_size=64, num_iterations=50, step_size=1.0):

    assert isinstance(model, models.Model)

    layer_output = model.get_layer(layer_name).output

    # Following is the loss function whose value we are to maximize.
    # Namely, starting from a blank image, we will proceed to doing
    # gradient ascent in input space in order to maximize the response
    # of this filter.
    loss_function = backend.mean(layer_output[:, :, :, filter_index])
    
    gradients = backend.gradients(loss_function, model.input)[0]
    # Gradient normalization trick:
    gradients /= (backend.sqrt(backend.mean(backend.square(gradients))) + 1e-4)

    func = backend.function([model.input], [loss_function, gradients])
    input_tensor = 128.0 + 20 * np.random.random((1, img_size, img_size, 3))
    
    for iteration in range(num_iterations):
        _, gradient_values = func([input_tensor])
        input_tensor += step_size * gradient_values
    
    img = to_valid_img(input_tensor[0])

    return img

    
def to_valid_img(arr):

    arr -= arr.mean()
    arr /= (arr.std() + 1e-4)
    arr *= 0.1
    
    arr += 0.5
    arr = np.clip(arr, 0, 1)

    arr *= 255
    arr = np.clip(arr, 0, 255).astype('uint8')

    return arr

    
def display_filters(model, layer_name, img_size=64,
    margin=5, grid_size=8, odir=getcwd()):

    assert isinstance(model, models.Model)

    grid = np.zeros((grid_size * img_size + (grid_size - 1) * margin,
                     grid_size * img_size + (grid_size - 1) * margin,
                     3))

    for row, column in product(range(grid_size), range(grid_size)):
        picture = maximally_responsive_pattern(
            model, layer_name,
            column + grid_size * row,
            img_size=img_size)
        
        column_begin = column * (img_size + margin)
        column_end = column_begin + img_size
        row_begin = row * (img_size + margin)
        row_end = row_begin + img_size
        
        grid[column_begin:column_end, row_begin:row_end, :] = picture
    
    plt.figure(figsize=(20, 20))
    plt.imshow(grid)
    with open(path.join(odir, layer_name + '.png'), 'w') as fh:
        plt.savefig(fh)
    plt.show()
    

def display_intermediate_activations(model, file_path, num_layers=8,
    height=150, width=150, odir=getcwd()):

    assert isinstance(model, models.Model)
    assert path.isfile(file_path)
    
    from keras.preprocessing import image

    img = image.load_img(file_path, target_size=(height, width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    
    plt.imshow(img[0])
    with open(path.join(odir, 'cat.png'), 'w') as fh:
        plt.savefig(fh)

    layer_names = list()
    layer_outputs = list()
    for layer in model.layers[:num_layers]:
        if isinstance(layer, layers.Conv2D):
            layer_names.append(layer.name)
        layer_outputs.append(layer.output)
        
    intermediate_activations_model = models.Model(
        input=model.input, output=layer_outputs)
    activations = intermediate_activations_model.predict(img)

    for layer_name, activation in zip(layer_names, activations):
        _, height, _, num_filters = activation.shape
        num_columns = num_filters / 16
        
        grid = np.zeros((num_columns * height, 16 * height))

        for column, row in product(range(num_columns), range(16)):
            picture = activation[0, :, :, row + 16 * column]
            
            picture = StandardScaler().fit_transform(picture)
            picture *= 64
            picture += 128
            picture = np.clip(picture, 0, 255).astype('uint8')
            
            grid[column * height:(column + 1) * height,
                 row * height:(row + 1) * height] = picture
        
        plt.figure(figsize=(grid.shape[1] / float(height),
            grid.shape[0] / float(height)))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(grid, aspect='auto', cmap='viridis')
        with open(path.join(odir, layer_name + '.png'), 'w') as fh:
            plt.savefig(fh)

    plt.show()


def trial_1(odir):
           
    print("Displaying the activations of every channel in every "
          "intermediate layer activation on a randomly-selected "
          "cat picture from the catsVsDogs test set:\n\n\n")
    
    model = models.load_model('catsVsDogs_small_convnet_experiment_1.h5')
    # Convnet trained on 2000 images of dogs and cats;
    # no pre-training involved
    model.summary()
    
    file_path = path.join(path.dirname(getcwd()),
        'data', 'catsVsDogs_small', 'test', 'cats',
        'cat.{}.jpg'.format(random.randint(1500, 1999)))
    
    display_intermediate_activations(
        model, file_path,
        odir=path.join(odir, 'intermediate_activations')
    )
           
           
def trial_2(odir):
           
    print("\n\n\nDisplaying the response patterns of the first 64 filters "
          "in the first layer of each convolution block of the VGG16 "
          "deep neural network architecture pre-trained on the "
          "ImageNet dataset:\n\n\n")
    
    model = vgg16.VGG16(include_top=False, weights='imagenet')
    model.summary()
    
    for i in range(1, 6):
        layer_name = 'block{}_conv1'.format(i)
        display_filters(
            model, layer_name,
            odir=path.join(odir, 'filter_patterns')
        )
           
           
def trial_3(odir):
           
    print("\n\n\nDisplaying a class activation map, i.e. a heatmap of 'class "
          "activation' over an input image that for a particular class "
          "indicates how important each location in that image is to that "
          "classifying that image or an object within as representative "
          "of that class. We are being tricky in submitting an image "
          "of a raven that the neural network finds ambiguous to classifiy "
          "(the top 2 predicted classes are 'vulture' and 'magpie').\n")

    file_path = path.join(path.dirname(getcwd()), 'data', 'raven.jpg')
    class_activation_heatmap(file_path, architecture='vgg16',
        odir=path.join(odir, 'class_activation_maps'))
   

def main():

    try:
        odir = path.join(path.dirname(getcwd()), 'output',
            'visualizing_convnets')
        makedirs(odir)
        for name in ('class_activation_maps', 'filter_patterns',
                     'intermediate_activations'):
            subdidr = path.join(odir, name)
            makedirs(subdir)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
        
    trial_1(odir)
    trial_2(odir)
    trial_3(odir)

    
if __name__ == '__main__':

    main()
