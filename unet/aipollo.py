import models
import utils
from datetime import datetime
import time
import data
import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from statistics import mean
import sklearn.metrics
from collections import defaultdict

IMG_HEIGHT = 256
IMG_WIDTH = 512
IMG_CHANNELS = 1
NUM_CLASSES = 158

log_path = 'C:/Users/simon/Coding/ML/aipollo/logs/'


class ShowPrediction(tf.keras.callbacks.Callback):
    def __init__(self, x, y, frequency = 10):
        self.x = x
        self.y = y
        self.counter = 0
        self.frequency = frequency

    def on_batch_end(self, epoch, logs={}):
        if self.counter % self.frequency == 0:
            utils.show_prediction(self.model, self.x, block=False)
        
        self.counter += 1

class SaveModelAfterBatch(tf.keras.callbacks.Callback):
    def __init__(self, frequency=50):
        self.frequency = frequency
        self.batch = 0

    def on_batch_end(self, epoch, logs={}):
        if self.batch % self.frequency == 0:
            logdir = log_path + datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H.%M.%S')
            os.makedirs(logdir)
            self.model.save(logdir)
        
        self.batch += 1

class PerClassMetric(tf.keras.callbacks.Callback):
    """
    Args:
        x: a 2D array
        y: a 2D array
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def on_batch_end(self, epoch, logs={}):
        utils.benchmark_model(self.model, [(self.x, self.y)])

def benchmark_model(num_instances=30):
    # Calculate precision and recall for all classes, averaged over all instances. Sort them by F1 score.
    instances = [data.get_random_instance(512, 1024) for x in range(num_instances)]
    model = models.get_pretrained_unet(instances[0][0].shape[0], instances[0][0].shape[1], NUM_CLASSES)
    utils.benchmark_model(model, instances)
    pass

def make_prediction():
    # Get image
    image, mask = data.get_random_instance(512, 1024)

    # Make sure image size is divisible by 16
    image = cv2.resize(image, (image.shape[1] + (16 - image.shape[1] % 16), image.shape[0] + (16 - image.shape[0] % 16)))
    mask = cv2.resize(mask, (mask.shape[1] + (16 - mask.shape[1] % 16), mask.shape[0] + (16 - mask.shape[0] % 16)))
    assert image.shape == mask.shape
    print(image.shape)

    # Get a model of the appropriate size
    model = models.get_unet(image.shape[0], image.shape[1], 1, NUM_CLASSES)
    trained_model = tf.keras.models.load_model('C:/Users/simon/Coding/ML/aipollo/logs/2020-03-07 16.45.22/')
    model.set_weights(trained_model.get_weights())

    # Show prediction
    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    utils.show_prediction(model, image, block=False)

    # Show mask
    mask = tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)
    print(mask.shape)
    mask = np.argmax(mask, axis=2)
    utils.show_mask(mask, 'mask', block=True)

    # Classify the image
    #print(image.shape)

    #prediction = model.predict(image)
    #prediction = np.argmax(prediction, axis=3)

    # Show prediction
    #prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])
    #print(prediction.shape)
    #cv2.imshow('Prediction', prediction)
    #cv2.waitKey(0)

def train_model(save=False):
    data_provider = data.DataProvider(IMG_HEIGHT, IMG_WIDTH, one_hot=False)
    dataset = tf.data.Dataset.from_generator(data_provider.yield_data, (tf.float32, tf.int32), ((IMG_HEIGHT, IMG_WIDTH, 1), (IMG_HEIGHT, IMG_WIDTH)))
    dataset = dataset.batch(4)

    first_batch_images, first_batch_masks = next(dataset.__iter__())
    debug_image, debug_mask = first_batch_images[0, :, :, :], first_batch_masks[0, :, :]
    debug_image = debug_image.numpy().reshape(IMG_HEIGHT, IMG_WIDTH)
    debug_mask = debug_mask.numpy().reshape(IMG_HEIGHT, IMG_WIDTH)

    #model = models.get_simple_cnn(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, data_provider.get_number_classes())
    model = models.get_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, data_provider.get_number_classes())
    callbacks = [PerClassMetric(debug_image, debug_mask), ShowPrediction(debug_image, debug_mask, frequency=1)]
    if save:
        callbacks.append(SaveModelAfterBatch(frequency=800))
    history = model.fit(dataset, callbacks=callbacks)

def custom_training():
    data_provider = data.DataProvider(IMG_HEIGHT, IMG_WIDTH, one_hot=False)
    dataset = tf.data.Dataset.from_generator(data_provider.yield_data, (tf.float32, tf.int32), ((IMG_HEIGHT, IMG_WIDTH, 1), (IMG_HEIGHT, IMG_WIDTH)))
    dataset = dataset.batch(1)

    debug_image, debug_mask = next(dataset.__iter__())
    debug_image = debug_image.numpy().reshape(IMG_HEIGHT, IMG_WIDTH)
    debug_mask = debug_mask.numpy().reshape(IMG_HEIGHT, IMG_WIDTH)

    model = models.get_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, data_provider.get_number_classes(), one_hot=False)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    for batch in dataset:
        x, y = batch[0], batch[1]
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True) # Or feed in the entire batch?

            loss = 0
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    true_class = y[0][i][j]
                    loss -= tf.math.log(y_pred[0][i][j][true_class])
            print(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        utils.show_prediction(model, debug_image)
        utils.benchmark_model(model, [(debug_image, debug_mask)])




#benchmark_model(num_instances=50)
#make_prediction()
train_model(save=False)
#custom_training()