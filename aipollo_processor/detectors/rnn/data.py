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
IMAGE_WIDTH = 640
NUM_CHANNELS = 1
SUBIMAGE_HEIGHT = 128
SUBIMAGE_WIDTH = 64

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

def generate_scores(num_scores=1024, length_range=(23,28), start_number=0):
    for i in tqdm.tqdm(range(num_scores)):
        number_of_symbols = random.randint(length_range)
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
        symbols_file = data_folder / (str(i + start_number) + '.txt')
        with open(symbols_file, 'w+', encoding='utf-8') as f:
            f.write(';'.join(symbols))
            
        lilypond_source = r'''
        \absolute {
        \set Score.timing = ##f
        \clef treble ''' + ' '.join(symbols) + ' }'
        print(lilypond_source)

        # Write Lilypond source file and translate it.
        lilypond_source_file = data_folder / (str(i + start_number) + '.ly')
        with open(lilypond_source_file, 'w+', encoding='utf-8') as f:
            f.write(lilypond_source)

        os.system(r'"' + lilypond_path + r'"' + ' --format=png --output=' + str(data_folder) + ' ' + str(data_folder / lilypond_source_file))


def get_autoencoder_dataset(batch_size=8, num_instances=None, val_proportion=0.05):
    if not num_instances:
        num_instances = max(int(file_name.split('.')[0]) for file_name in os.listdir(data_folder) if file_name.split('.')[0].isnumeric())

    print(f'The dataset contains {num_instances} instances.')

    path_ds = tf.data.Dataset.from_tensor_slices([str(data_folder / (str(i) + '.png')) for i in range(num_instances)])
    image_ds = path_ds.map(lambda path: _load_and_preprocess_image(path)[:SUBIMAGE_HEIGHT, 100:SUBIMAGE_WIDTH+100, :])
    all_ds = tf.data.Dataset.zip((image_ds, image_ds))

    val_ds = all_ds.take(int(val_proportion * num_instances))
    train_ds = all_ds.skip(int(val_proportion * num_instances))
    del all_ds

    train_ds = train_ds.shuffle(128)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.shuffle(128)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds

def get_train_validation(batch_size=8, num_instances=None, val_proportion=0.05, pad_to_length=None, with_shifted=False, one_hot=False):
    if not num_instances:
        num_instances = max(int(file_name.split('.')[0]) for file_name in os.listdir(data_folder) if file_name.split('.')[0].isnumeric())

    print(f'The dataset will contain {num_instances} instances.')

    path_ds = tf.data.Dataset.from_tensor_slices([str(data_folder / (str(i) + '.png')) for i in range(num_instances)])
    image_ds = path_ds.map(lambda path: _load_and_preprocess_image(path)[:SUBIMAGE_HEIGHT, :, :])

    all_train_labels = []
    all_train_labels_shifted = []
    for i in range(num_instances):
        with open(data_folder / (str(i) + '.txt'), 'r', encoding='utf-8') as f:
            symbols = f.read().split(';')
            symbols_as_numbers = [symbol_to_number(symbol) for symbol in symbols]
            symbols_as_numbers_shifted = [symbol_to_number('START')] + symbols_as_numbers[:-1]

            if pad_to_length:
                if len(symbols_as_numbers) > pad_to_length:
                    raise ValueError

                symbols_as_numbers.extend([symbol_to_number('STOP')] * (pad_to_length - len(symbols_as_numbers)))
                symbols_as_numbers_shifted.extend([symbol_to_number('STOP')] * (pad_to_length - len(symbols_as_numbers_shifted)))
                assert len(symbols_as_numbers) == pad_to_length

            if one_hot:
                symbols_as_numbers = tf.keras.utils.to_categorical(symbols_as_numbers, num_symbols)
                symbols_as_numbers_shifted = tf.keras.utils.to_categorical(symbols_as_numbers_shifted, num_symbols)

            all_train_labels.append(symbols_as_numbers)
            all_train_labels_shifted.append(symbols_as_numbers_shifted)

    target_ds = tf.data.Dataset.from_generator(lambda: all_train_labels, tf.int64)
    target_shifted_ds = tf.data.Dataset.from_generator(lambda: all_train_labels_shifted, tf.int64)

    #create (image, label) zip to iterate over
    if with_shifted:
        data_label_ds = tf.data.Dataset.zip((tf.data.Dataset.zip((image_ds, target_shifted_ds)), target_ds))
    else:
        data_label_ds = tf.data.Dataset.zip((image_ds, target_ds))

    #Generate a validation set
    val_ds = data_label_ds.take(int(val_proportion * num_instances))
    train_ds = data_label_ds.skip(int(val_proportion * num_instances))

    #training data producer
    train_ds = train_ds.shuffle(128)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #validation data producer
    val_ds = val_ds.shuffle(128)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds


def _load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = image[:IMAGE_HEIGHT, 50:-20, :]
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image /= 255.0

    return image