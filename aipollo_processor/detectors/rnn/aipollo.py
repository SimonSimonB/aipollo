import tensorflow as tf
import numpy as np
import models
import data
import cv2
import pathlib

checkpoints_folder = pathlib.Path(r'C:\Users\simon\Coding\ML\aipollo\rnn\checkpoints')

class ShowReconstruction(tf.keras.callbacks.Callback):
    def __init__(self, x, frequency = 10):
        self._x = x
        self._counter = 0
        self._frequency = frequency

    def on_batch_end(self, epoch, logs={}):
        if self._counter % self._frequency == 0:
            reconstruction = np.asarray(self.model.predict(tf.convert_to_tensor([self._x])))
            reconstruction = reconstruction[0]

            cv2.imshow('original', self._x[:, :, 0])
            cv2.imshow('reconstruction', reconstruction[:, :, 0])
            cv2.waitKey(1)
            #matplotlib.pyplot.imshow(reconstruction[:, :, 0], cmap='gray')
            #matplotlib.pyplot.imshow(self._x[:, :, 0], cmap='gray')
            #matplotlib.pyplot.show()

        self._counter += 1


def train_autoencoder():
    model = models.ConvNetAutoEncoder(subimage_height=data.SUBIMAGE_HEIGHT, subimage_width=data.SUBIMAGE_WIDTH, num_channels=data.NUM_CHANNELS, num_subimage_features=256)
    train_ds, val_ds = data.get_autoencoder_dataset(batch_size=16)

    # Debug
    example = next(val_ds.take(1).as_numpy_iterator())[0][0, :, :, :]

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.MeanSquaredError())
    model.fit(
        train_ds,
        epochs=256, 
        callbacks=[ShowReconstruction(example, frequency=8), tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoints_folder / 'cp-{epoch:04d}.ckpt'), save_weights_only=True)],
        validation_data=val_ds)
    print('Done')

def train_classifier():
    model = models.get_fixed_size_crnn(
        output_alphabet_size=data.num_symbols, 
        max_seq_length=30,
        image_height=data.IMAGE_HEIGHT, 
        image_width=data.IMAGE_WIDTH, 
        subimage_height=data.SUBIMAGE_HEIGHT, 
        subimage_width=data.SUBIMAGE_WIDTH, 
        num_channels=data.NUM_CHANNELS
    )
    train, val = data.get_train_validation(num_instances=64, batch_size=4, pad_to_length=30, with_shifted=True, one_hot=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    batch = next(val.take(1).as_numpy_iterator())[0]
    x, target_shifted = batch[0][0], batch[1][0]
    f = model.predict([tf.convert_to_tensor([x]), tf.convert_to_tensor([target_shifted])])

    model.fit(train)


def train_classifier_manual():
    train, val = data.get_train_validation(num_instances=None, batch_size=1)
    model = models.CRNN(data.symbol_to_number('START'), data.num_symbols, data.IMAGE_HEIGHT, data.IMAGE_WIDTH, data.SUBIMAGE_HEIGHT, data.SUBIMAGE_WIDTH, data.NUM_CHANNELS)

    #model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.CategoricalCrossentropy())
    #model.fit(train)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    ctr = 0
    for instance in train:
        x, y = instance[0][0], instance[1][0] 
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

            ctr += 1
            print(ctr)
            print(loss)
            print(str(y_with_stop.tolist()))
            print(debug_prediction)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == '__main__':
    #data.generate_scores(2**14, start_number=2048)
    #exit()
    #train_autoencoder()
    #exit()
    train_classifier()
    