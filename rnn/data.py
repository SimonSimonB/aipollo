import random
import numpy as np
import tensorflow as tf
import os
import pathlib
import itertools
import tqdm
import matplotlib.pyplot

lilypond_path = r'C:\Program Files (x86)\LilyPond\usr\bin\lilypond.exe'
data_folder = pathlib.Path(r'C:\Users\simon\Coding\ML\aipollo\rnn\data')

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 600
NUM_CHANNELS = 3
SUBIMAGE_HEIGHT = 128
SUBIMAGE_WIDTH = 128

symbols_with_duration = [x[0] + x[1] for x in list(itertools.product(['c\'', 'd\'', 'e\'', 'f\'' , 'g\'', 'a\'', 'b\'', 'r'], ['1', '2', '4', '8']))]
symbols_without_duration = [r'\bar "|"']
num_symbols = len(symbols_with_duration) + len(symbols_without_duration) + 2

def symbol_to_number(symbol):
    if symbol == 'START':
        return 0
    elif symbol == 'STOP':
        return 1
    elif symbol in symbols_without_duration:
        return symbols_without_duration.index(symbol) + 2
    elif symbol in symbols_with_duration:
        return len(symbols_without_duration) + 2 + symbols_with_duration.index(symbol)
    else:
        raise Exception

def generate_scores(num_scores=5):
    for i in tqdm.tqdm(range(num_scores)):
        number_of_symbols = random.randint(23,28)
        symbols = []
        no_barline_next = True
        for _ in range(number_of_symbols):
            if random.random() < 0.5 and not no_barline_next:
                symbols.append(random.choice(symbols_without_duration))
                no_barline_next = True
            else:
                symbols.append(random.choice(symbols_with_duration))
                no_barline_next = False
        
        
        # Write symbols to file. These will be the ground truth labels.
        symbols_file = data_folder / (str(i) + '.txt')
        with open(symbols_file, 'w+', encoding='utf-8') as f:
            f.write(';'.join(symbols))
            
        lilypond_source = r'''
        \absolute {
        \set Score.timing = ##f
        \clef treble ''' + ' '.join(symbols) + ' }'
        print(lilypond_source)

        # Write Lilypond source file and translate it.
        lilypond_source_file = data_folder / (str(i) + '.ly')
        with open(lilypond_source_file, 'w+', encoding='utf-8') as f:
            f.write(lilypond_source)

        os.system(r'"' + lilypond_path + r'"' + ' --format=png --output=' + str(data_folder) + ' ' + str(data_folder / lilypond_source_file))


def get_autoencoder_dataset(batch_size=8, num_instances=2048):
    path_ds = tf.data.Dataset.from_tensor_slices([str(data_folder / (str(i) + '.png')) for i in range(num_instances)])
    image_ds = path_ds.map(lambda path: _load_and_preprocess_image(path)[:SUBIMAGE_HEIGHT, 100:SUBIMAGE_WIDTH+100, :])
    #image_ds = path_ds.map(lambda path: _load_and_preprocess_image(path))
    autoencoder_ds = tf.data.Dataset.zip((image_ds, image_ds))

    autoencoder_ds = autoencoder_ds.shuffle(128)
    autoencoder_ds = autoencoder_ds.batch(batch_size)
    autoencoder_ds = autoencoder_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return autoencoder_ds

def get_train_validation(num_instances=5, batch_size=2):
    all_train_labels = []
    for i in range(num_instances):
        with open(data_folder / (str(i) + '.txt'), 'r', encoding='utf-8') as f:
            symbols = f.read().split(';')
            symbols_as_numbers = [symbol_to_number(symbol) for symbol in symbols]
            all_train_labels.append(symbols_as_numbers)

    path_ds = tf.data.Dataset.from_tensor_slices([str(data_folder / (str(i) + '.png')) for i in range(num_instances)])
    image_ds = path_ds.map(_load_and_preprocess_image)
    label_ds = tf.data.Dataset.from_generator(lambda: all_train_labels, tf.int64)

    #create (image, label) zip to iterate over
    data_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    #Generate a validation set
    VAL_COUNT = 1
    val_label_ds = data_label_ds.take(VAL_COUNT)
    train_label_ds = data_label_ds.skip(VAL_COUNT)

    #training data producer
    tds = train_label_ds.shuffle(128)
    tds = tds.repeat()
    #tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #tds = tds.cache(filename=’./save/’)

    #validation data producer
    vds = val_label_ds.shuffle(VAL_COUNT)
    vds = vds.repeat()
    #vds = vds.batch(batch_size)
    vds = vds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return tds, vds

def _load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image[:IMAGE_HEIGHT, 50:-20, :]
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image /= 255.0

    return image