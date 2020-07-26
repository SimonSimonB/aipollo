import random
import numpy as np
import tensorflow as tf
import os
import pathlib
import itertools
import tqdm

lilypond_path = r'C:\Program Files (x86)\LilyPond\usr\bin\lilypond.exe'
data_folder = pathlib.Path(r'C:\Users\simon\Coding\ML\aipollo\rnn\data')

HEIGHT = 100
WIDTH = 400
NUM_CHANNELS = 3
SUBIMAGE_HEIGHT = 100
SUBIMAGE_WIDTH = 100

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

class CRNN(tf.keras.Model):
    def __init__(self, num_subimage_features=128, latent_dim=32, output_alphabet_size=num_symbols):
        super(tf.keras.Model, self).__init__()
        self._conv_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), input_shape=(SUBIMAGE_HEIGHT, SUBIMAGE_WIDTH, NUM_CHANNELS)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_subimage_features)
        ])
        print(self._conv_net.summary())

        self._encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
        self._decoder = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        self._decoder_dense = tf.keras.layers.Dense(output_alphabet_size, activation='softmax')


    # Takes a single element (not a batch), with each element consisting of a vector of subimages and the target output sequence
    def call(self, inputs):
        image, target_sequence = inputs
        subimages = []
        for start_x in range(0, WIDTH, SUBIMAGE_WIDTH):
            assert image.dtype == tf.float32
            subimage = np.ones(shape=(HEIGHT, SUBIMAGE_WIDTH, NUM_CHANNELS), dtype=np.float32)
            subimage[:, 0:min(SUBIMAGE_WIDTH, WIDTH-start_x), :] = image[:, start_x:min(start_x + SUBIMAGE_WIDTH, WIDTH), :]
            subimages.append(subimage)

        conv_net_outputs = self._conv_net(tf.convert_to_tensor(subimages))

        target_sequence_shifted = np.insert(target_sequence, 0, symbol_to_number('START'))
        target_sequence_shifted_one_hot = tf.keras.utils.to_categorical(target_sequence_shifted, num_symbols)

        _, state_h, state_c = self._encoder(tf.convert_to_tensor([conv_net_outputs]))
        encoder_states = [state_h, state_c]
        decoder_outputs, _, _ = self._decoder(tf.convert_to_tensor([target_sequence_shifted_one_hot]), initial_state=encoder_states)
        decoder_outputs = self._decoder_dense(decoder_outputs)
        decoder_outputs = decoder_outputs[0]

        return decoder_outputs
 

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
    tds = train_label_ds.shuffle(VAL_COUNT)
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
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image /= 255.0
    return image

if __name__ == '__main__':
    #generate_scores(128)
    train, val = get_train_validation(num_instances=128)
    model = CRNN()
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #model.fit(train)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    for instance in train:
        x, y = instance[0], instance[1]
        with tf.GradientTape() as tape:
            y_pred = model((x, y), training=True)
            y_with_stop = np.append(y, symbol_to_number('STOP'))
            del y

            assert y_with_stop.shape[0] == y_pred.shape[0]
            seq_length = y_with_stop.shape[0]

            loss = 0
            debug_prediction = []
            for step in range(seq_length):
                true_class = y_with_stop[step]
                loss -= tf.math.log(y_pred[step][true_class])
                debug_prediction.append(np.argmax(y_pred[step]))
            print(loss)
            print(str(y_with_stop.tolist()))
            print(debug_prediction)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
