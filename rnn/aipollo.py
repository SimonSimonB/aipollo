import tensorflow as tf
import numpy as np
import models
import data
import matplotlib.pyplot



def train_autoencoder():
    model = models.ConvNetAutoEncoder(subimage_height=data.SUBIMAGE_HEIGHT, subimage_width=data.SUBIMAGE_WIDTH, num_channels=data.NUM_CHANNELS, num_subimage_features=256)
    autoencoder_dataset = data.get_autoencoder_dataset()

    # Debug
    example = next(autoencoder_dataset.take(1).as_numpy_iterator())[0][0, :, :, :]
    matplotlib.pyplot.imshow(example, cmap='gray')
    matplotlib.pyplot.show()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.MeanSquaredError())
    model.fit(autoencoder_dataset, epochs=256)
    print('Done')

if __name__ == '__main__':
    #data.generate_scores(2048)
    #exit()
    train_autoencoder()
    exit()
    train, val = data.get_train_validation(num_instances=128)
    model = models.CRNN(data.symbol_to_number('START'), data.num_symbols, data.IMAGE_HEIGHT, data.IMAGE_WIDTH, data.SUBIMAGE_HEIGHT, data.SUBIMAGE_WIDTH, data.NUM_CHANNELS)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #model.fit(train)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    for instance in train:
        x, y = instance[0], instance[1]
        with tf.GradientTape() as tape:
            y_pred = model((x, y), training=True)
            y_with_stop = np.append(y, data.symbol_to_number('STOP'))
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
