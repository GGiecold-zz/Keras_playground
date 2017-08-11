#!/usr/bin/env python


from keras import layers, models
from keras.optimizers import RMSProp
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


def vectorize(sequences, dim=10000):
  res = np.zeros((len(sequences), dim), dtype='float32')
  
  for i, seq in enumerate(sequences):
    res[i, seq] = 1.0
    
  return res


def build_model(train_labels, num_words=10000):
  model = models.Sequential()
  
  model.add(layers.Dense(64, activation='relu', input_shape=(num_words,)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(max(train_labels) + 1, activation='softmax'))
  
  return model
  

def plot_loss_and_accuracy(history):
  dico = history.history
  
  training_losses = dico['loss']
  validation_losses = dico['val_loss']
  
  epochs = range(1, len(training_losses) + 1)
  
  plt.plot(epochs, training_losses, 'b+')
  plt.plot(epochs, validation_losses, 'r+')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  
  plt.show()
  sleep(10)
  plt.clf()
  
  training_accuracies = dico['binary_acc']
  validation_accuracies = dico['val_binary_acc']
  
  plt.plot(epochs, training_accuracies, 'bo')
  plt.plot(epochs, validation_accuracies, 'ro')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  
  plt.show()
  sleep(10)
  plt.clf()
  
  
int main():

  from keras.datasets import reuters

  (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

  print("First sample of the Reuters training dataset:\n", train_data[0])

  word_index = reuters.get_word_index()
  inverse_word_index = dict((v, k) for k, v in word_index.iteritems())

  news_article_0 = ' '.join([inverse_word_index.get(i - 3, '?') for i in train_data[0]])
  print("Corresponding news article:", news_article_0)

  train_data = vectorize(train_data)
  test_data = vectorize(test_data)
  
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)
  
  model = build_model(train_labels)
  model.compile(optimizer=RMSProp(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  
  validation_data = train_data[:1000]
  validation_labels = train_labels[:1000]
  
  partial_train_data = train_data[1000:]
  partial_train_labels = train_labels[1000:]
  
  history = model.fit(partial_train_data, partial_train_labels, 
                      epochs=20, batch_size=512,
                      validation_data=(validation_data, validation_labels))
                      
  plot_loss_and_accuracy(history)
  
  model = build_model(train_labels)
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_data, train_labels, epochs=9, batch_size=512, validation_data=(test_data, test_labels))
  
  evaluation = model.evaluate(test_data, test_labels)
  print("Loss on test data:{}\nAccuracy on test data: {}\n".format(evaluation))
  
  predictions = model.predict(test_data)
  print("Topic with highest probability for the first news article in the test dataset:\n", 
        np.argmax(predictions[0]))
  

if __name__ == '__main__':

  main()
  
