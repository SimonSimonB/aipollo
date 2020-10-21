import tensorflow as tf
import numpy as np

#import tf_crnn.config
#def get_solivr_crnn():
#    params = tf_crnn.config.Params()



class ConvNetAutoEncoder(tf.keras.Sequential):
    def __init__(self, subimage_height, subimage_width, num_channels, num_subimage_features=128):
        super(ConvNetAutoEncoder, self).__init__([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), input_shape=(subimage_height, subimage_width, num_channels), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_subimage_features),
            tf.keras.layers.Dense(int(subimage_height/8) * int(subimage_width/8) * 64),
            tf.keras.layers.Reshape(target_shape=(int(subimage_height/8), int(subimage_width/8), 64)), # maybe flip?
            tf.keras.layers.UpSampling2D((2,2)),
            tf.keras.layers.Conv2DTranspose(32, (2, 2), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2,2)),
            tf.keras.layers.Conv2DTranspose(16, (2, 2), activation='relu', padding='same'), 
            tf.keras.layers.UpSampling2D((2,2)),
            tf.keras.layers.Conv2DTranspose(num_channels, (1, 1), activation='relu', padding='same')
        ])
        print(self.summary())

def get_fixed_size_crnn(output_alphabet_size, max_seq_length, image_height, image_width, subimage_height, subimage_width, num_channels, num_subimage_features=128, rnn_latent_dim=128):
    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), input_shape=(subimage_height, subimage_width, num_channels), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_subimage_features),
    ])

    encoder = tf.keras.layers.LSTM(rnn_latent_dim, return_state=True)
    decoder = tf.keras.layers.LSTM(rnn_latent_dim, return_sequences=True, return_state=True)
    dense = tf.keras.layers.Dense(output_alphabet_size, activation='softmax')

    image_input = tf.keras.Input(shape=(image_height, image_width, num_channels))
    target_sequence_shifted_by_one_input = tf.keras.Input(shape=(max_seq_length, output_alphabet_size))

    # Split image into subimages
    if not image_width % subimage_width == 0:
        raise ValueError(f'Image width {image_width} is not divisible by {subimage_width}')

    subimages = tf.keras.layers.Lambda(lambda full_image: tf.split(full_image, axis=2, num_or_size_splits=int(image_width // subimage_width)))(image_input)

    #subimage = []
    #for i in range(int(image_width // subimage_width)):
    #    subimage_layer = tf.keras.layers.Lambda(lambda full_image: full_image[:, i*subimage_width:(i+1)*subimage_width, :])(inputs)
    #    subimage.append(tf.keras.layers.Lambda(lambda full_image: full_image[:, i*subimage_width:(i+1)*subimage_width, :])(inputs))
    #subimages = tf.keras.layers.concatenate(subimage)

    # Feed subimages through CNN
    subimages_latent = []
    for i in range(int(image_width // subimage_width)):
        subimages_latent.append(cnn(subimages[i]))
    
    subimages_latent = tf.keras.layers.Lambda(lambda subimages_latent: tf.stack(subimages_latent, axis=2))(subimages_latent)
    _, state_h, state_c = encoder(subimages_latent)
    outputs = decoder(target_sequence_shifted_by_one_input, initial_state=[state_h, state_c])[0]
    outputs = dense(outputs)

    model = tf.keras.Model(inputs=[image_input, target_sequence_shifted_by_one_input], outputs=outputs)

    return model

class CRNN(tf.keras.Model):
    def __init__(self, start_number_code, output_alphabet_size, image_height, image_width, subimage_height, subimage_width, num_channels, num_subimage_features=128, rnn_latent_dim=128):
        super(tf.keras.Model, self).__init__()

        self._image_height = image_height
        self._image_width = image_width
        self._subimage_height = subimage_height
        self._subimage_width = subimage_width
        self._num_channels = num_channels
        self._start_number_code = start_number_code
        self._output_alphabet_size = output_alphabet_size

        self._conv_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), input_shape=(subimage_height, subimage_width, num_channels), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_subimage_features),
        ])
        print(self._conv_net.summary())

        self._encoder = tf.keras.layers.LSTM(rnn_latent_dim, return_state=True)
        self._decoder = tf.keras.layers.LSTM(rnn_latent_dim, return_sequences=True, return_state=True)
        self._decoder_dense = tf.keras.layers.Dense(output_alphabet_size, activation='softmax')


    # Takes a single element (not a batch), with each element consisting of a vector of subimages and the target output sequence
    def call(self, inputs):
        image, target_sequence = inputs
        subimages = []
        for start_x in range(0, self._image_width, self._subimage_width):
            assert image.dtype == tf.float32
            subimage = np.ones(shape=(self._image_height, self._subimage_width, self._num_channels), dtype=np.float32)
            subimage[:, 0:min(self._subimage_width, self._image_width-start_x), :] = image[:, start_x:min(start_x + self._subimage_width, self._image_width), :]
            subimages.append(subimage)

        conv_net_outputs = self._conv_net(tf.convert_to_tensor(subimages))

        target_sequence_shifted = np.insert(target_sequence, 0, self._start_number_code)
        target_sequence_shifted_one_hot = tf.keras.utils.to_categorical(target_sequence_shifted, self._output_alphabet_size)

        _, state_h, state_c = self._encoder(tf.convert_to_tensor([conv_net_outputs]))
        encoder_states = [state_h, state_c]
        decoder_outputs, _, _ = self._decoder(tf.convert_to_tensor([target_sequence_shifted_one_hot]), initial_state=encoder_states)
        decoder_outputs = self._decoder_dense(decoder_outputs)
        decoder_outputs = decoder_outputs[0]

        return decoder_outputs