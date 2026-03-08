# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Dropout,
    Dense,
    Flatten,
    Reshape,
    Conv2DTranspose,
    BatchNormalization,
    LayerNormalization,
    SpatialDropout2D,
    UpSampling2D,
)
try:
    # TF >= 2.6 exposes preprocessing layers directly under keras.layers.
    from tensorflow.keras.layers import Rescaling
except ImportError:
    # Backward compatibility for older TF releases.
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import commons
import dataset_manager as dsm

batch_size = 32
epochs = 300
patience = 10
truly_training_percentage = 0.80


def conv_block(entry, layers, filters, dropout, first_block=False):
    conv = None
    for i in range(layers):
        if first_block:
            conv = Conv2D(
                kernel_size=3,
                padding='same',
                activation='relu',
                filters=filters,
                input_shape=(dsm.columns, dsm.rows, 1),
            )(entry)
            first_block = False
        else:
            conv = Conv2D(
                kernel_size=3, padding='same', activation='relu', filters=filters
            )(entry)
        entry = BatchNormalization()(conv)
    pool = MaxPool2D(pool_size=2, strides=2, padding='same')(entry)
    drop = SpatialDropout2D(dropout)(pool)
    return drop


# The number of layers defined in get_encoder.
encoder_nlayers = 40


def get_encoder(domain):
    dropout = 0.5
    input_data = Input(shape=(dsm.columns, dsm.rows, 1))
    filters = domain // 16
    output = conv_block(input_data, 2, filters, dropout, first_block=True)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 2, filters, dropout)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 3, filters, dropout)
    output = Flatten()(output)
    output = LayerNormalization(name='encoded')(output)
    return input_data, output


def get_decoder(domain):
    input_mem = Input(shape=(domain,))
    width = dsm.columns // 4
    filters = domain // 4
    dense = Dense(width * width * filters, activation='relu', input_shape=(domain,))(
        input_mem
    )
    output = Reshape((width, width, filters))(dense)
    dropout = 0.4
    for i in range(2):
        trans = Conv2DTranspose(
            kernel_size=3, strides=2, padding='same', activation='relu', filters=filters
        )(output)
        output = SpatialDropout2D(dropout)(trans)
        dropout /= 2.0
        filters = filters // 4
        output = BatchNormalization()(output)
    output = Conv2DTranspose(
        filters=filters, kernel_size=3, strides=1, activation='sigmoid', padding='same'
    )(output)
    output_img = Rescaling(255.0, name='decoded')(output)
    return input_mem, output_img


# The number of layers defined in get_classifier.
classifier_nlayers = 6


def get_classifier(domain):
    input_mem = Input(shape=(domain,))
    dense = Dense(domain, activation='relu', input_shape=(domain,))(input_mem)
    drop = Dropout(0.4)(dense)
    dense = Dense(domain, activation='relu')(drop)
    drop = Dropout(0.4)(dense)
    classification = Dense(commons.n_labels, activation='softmax', name='classified')(
        drop
    )
    return input_mem, classification


class EarlyStopping(Callback):
    """Stop training when the loss gets lower than val_loss.

    Arguments:
        patience: Number of epochs to wait after condition has been hit.
        After this number of no reversal, training stops.
        It starts working after 10% of epochs have taken place.
    """

    def __init__(self):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.prev_val_loss = float('inf')
        self.prev_val_accuracy = 0.0
        self.prev_val_rmse = float('inf')

        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = min(epochs // 20, 3)
        self.wait = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('classifier_accuracy')
        val_accuracy = logs.get('val_classifier_accuracy')
        rmse = logs.get('decoder_root_mean_squared_error')
        val_rmse = logs.get('val_decoder_root_mean_squared_error')

        if epoch < self.start:
            self.best_weights = self.model.get_weights()
        elif (loss < val_loss) or (accuracy > val_accuracy) or (rmse < val_rmse):
            self.wait += 1
        elif val_accuracy > self.prev_val_accuracy:
            self.wait = 0
            self.prev_val_accuracy = val_accuracy
            self.best_weights = self.model.get_weights()
        elif val_rmse < self.prev_val_rmse:
            self.wait = 0
            self.prev_val_rmse = val_rmse
            self.best_weights = self.model.get_weights()
        elif val_loss < self.prev_val_loss:
            self.wait = 0
            self.prev_val_loss = val_loss
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        print(f'Epochs waiting: {self.wait}')
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def train_network(ds, prefix, es):
    confusion_matrix = np.zeros((commons.n_labels, commons.n_labels))
    histories = []
    for fold in range(commons.n_folds):
        training_data, training_labels = dsm.get_training(ds, fold)
        testing_data, testing_labels = dsm.get_testing(ds, fold)
        truly_training = int(len(training_labels) * truly_training_percentage)
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)

        rmse = tf.keras.metrics.RootMeanSquaredError()
        input_data = Input(shape=(dsm.columns, dsm.rows, 1))
        domain = commons.domain(ds)
        input_enc, encoded = get_encoder(domain)
        encoder = Model(input_enc, encoded, name='encoder')
        encoder.compile(optimizer='adam')
        encoder.summary()
        input_cla, classified = get_classifier(domain)
        classifier = Model(input_cla, classified, name='classifier')
        classifier.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics='accuracy'
        )
        classifier.summary()
        input_dec, decoded = get_decoder(domain)
        decoder = Model(input_dec, decoded, name='decoder')
        decoder.compile(optimizer='adam', loss='mean_squared_error', metrics=rmse)
        decoder.summary()
        encoded = encoder(input_data)
        decoded = decoder(encoded)
        classified = classifier(encoded)
        full_classifier = Model(
            inputs=input_data, outputs=classified, name='full_classifier'
        )
        full_classifier.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics='accuracy'
        )
        autoencoder = Model(inputs=input_data, outputs=decoded, name='autoencoder')
        autoencoder.compile(loss='huber', optimizer='adam', metrics=rmse)

        model = Model(inputs=input_data, outputs=[classified, decoded])
        model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer='adam',
            metrics={'classifier': 'accuracy', 'decoder': rmse},
        )
        model.summary()
        history = model.fit(
            training_data,
            (training_labels, training_data),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                validation_data,
                {'classifier': validation_labels, 'decoder': validation_data},
            ),
            callbacks=[EarlyStopping()],
            verbose=2,
        )
        histories.append(history)
        history = full_classifier.evaluate(
            testing_data, testing_labels, return_dict=True
        )
        histories.append(history)
        predicted_labels = np.argmax(full_classifier.predict(testing_data), axis=1)
        confusion_matrix += tf.math.confusion_matrix(
            np.argmax(testing_labels, axis=1),
            predicted_labels,
            num_classes=commons.n_labels,
        )
        history = autoencoder.evaluate(testing_data, testing_data, return_dict=True)
        histories.append(history)
        encoder.save(commons.encoder_filename(prefix, es, fold))
        decoder.save(commons.decoder_filename(prefix, es, fold))
        classifier.save(commons.classifier_filename(prefix, es, fold))
        prediction_prefix = commons.classification_name(ds, es)
        prediction_filename = commons.data_filename(prediction_prefix, es, fold)
        np.save(prediction_filename, predicted_labels)
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1, 1)
    return histories, confusion_matrix / totals


def obtain_features(ds, model_prefix, features_prefix, labels_prefix, data_prefix, es):
    """Generate features for sound segments, corresponding to phonemes.

    Uses the previously trained neural networks for generating the features.
    """
    for fold in range(commons.n_folds):
        # Load de encoder
        filename = commons.encoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(filename)
        model.summary()

        training_data, training_labels = dsm.get_training(ds, fold)
        filling_data, filling_labels = dsm.get_filling(ds, fold)
        testing_data, testing_labels = dsm.get_testing(ds, fold)
        noised_data, noised_labels = dsm.get_testing(ds, fold, noised=True)
        settings = [
            (training_data, training_labels, commons.training_suffix),
            (filling_data, filling_labels, commons.filling_suffix),
            (testing_data, testing_labels, commons.testing_suffix),
            (noised_data, noised_labels, commons.noised_suffix),
        ]
        # Only testing data is saved, to recover the original images.
        save_on = 2
        for i, s in enumerate(settings):
            data = s[0]
            labels = s[1]
            suffix = s[2]
            features = model.predict(data)
            features_filename = commons.data_filename(
                features_prefix + suffix, es, fold
            )
            labels_filename = commons.data_filename(labels_prefix + suffix, es, fold)
            np.save(features_filename, features)
            np.save(labels_filename, labels)
            if i == save_on:
                data_filename = commons.data_filename(data_prefix + suffix, es, fold)
                np.save(data_filename, data)
