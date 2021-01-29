import tensorflow as tf
import numpy as np
import argparse
import datetime
from pathlib import Path
import shutil
import os
import sys
from tensorflow_exps import log_utils


class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # TODO add parameter for accuracy threshold
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


class TF(object):
    def __init__(self, file_info, params):
        self.file_info = file_info
        self.params = params

        self.assert_fileinfo()
        self.assert_params()

        self.x_train, self.y_train = np.array, np.array
        self.x_test, self.y_test = np.array, np.array
        # self.data_stats()
        self.model = None
        self.base_stats = ''
        self.log_dir = self.file_info['save_loc']

    def init(self):
        self.read_data()
        self.preprocess_data()

    def assert_fileinfo(self):
        assert 'save_loc' in self.file_info.keys(), 'save_loc in file_info is compulsory'

    def assert_params(self):
        # TODO check if multiple callbacks can be used
        if 'callback' not in self.params.keys():
            self.params['callback'] = None

        params_default = {'training_data': 'fashion_mnist', 'normalise_method': 'pixel',
                          'tf_compile': {'optimizer': 'sgd', 'loss': 'sparse_categorical_crossentropy',
                                         'metrics': ['accuracy'], 'epochs': 10},
                          'mode': 'basic_dense'}

        if 'normalise_method' not in self.params.keys():
            self.params['normalise_method'] = params_default['normalise_method']
        if 'training_data' not in self.params.keys():
            self.params['training_data'] = params_default['training_data']
        if 'mode' not in self.params.keys():
            self.params['mode'] = params_default['mode']
        if 'tf_compile' not in self.params.keys():
            self.params['tf_compile'] = params_default['tf_compile']
        if 'optimizer' not in self.params['tf_compile'].keys():
            self.params['tf_compile']['optimizer'] = params_default['tf_compile']['optimizer']
        if 'loss' not in self.params['tf_compile'].keys():
            self.params['tf_compile']['loss'] = params_default['tf_compile']['loss']
        if 'metrics' not in self.params['tf_compile'].keys():
            self.params['tf_compile']['metrics'] = params_default['tf_compile']['metrics']
        if 'epochs' not in self.params['tf_compile'].keys():
            self.params['tf_compile']['epochs'] = params_default['tf_compile']['epochs']

        if 'create_unique' in self.params.keys():
            assert self.params['create_unique'] in [True, False]
        else:
            self.params['create_unique'] = False

        if self.params['create_unique']:
            self.params['log_dir_suffix'] = '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.params['log_dir_suffix'] = ''

    def read_data(self):
        if self.params['training_data'] == 'fashion_mnist':
            self.base_stats += 'reading fashion_mnist \n'
            fmnist = tf.keras.datasets.fashion_mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fmnist.load_data()
            self.base_stats += 'shapes: \n\t x_train: {0} \n\t y_train: {1} \n\t x_test: {2} \n\t y_test: {3} \n'.format(
                np.shape(self.x_train), np.shape(self.x_train), np.shape(self.x_test), np.shape(self.y_test))
        elif self.params['training_data'] == 'mnist':
            self.base_stats += 'reading mnist \n'
            mnist = tf.keras.datasets.mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            self.base_stats += 'shapes: \n\t x_train: {0} \n\t y_train: {1} \n\t x_test: {2} \n\t y_test: {3} \n'.format(
                np.shape(self.x_train), np.shape(self.x_train), np.shape(self.x_test), np.shape(self.y_test))
        else:
            raise AssertionError('available training dataset options: mnist/fashion_mnist')

    def preprocess_data(self):
        if self.params['normalise_method'] == 'pixel':
            self.base_stats += 'normalising by 255.0\n'
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        elif self.params['normalise_method'] is None:
            self.base_stats += 'not normalising the input data, neural nets might not converge\n'
            pass
        else:
            raise NotImplementedError

    def build(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(input_shape=(np.shape(self.x_train)[1], np.shape(self.x_train)[2])),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        # TODO check binary_cross_entropy for binary classification
        # TODO check RMSProp as optimizer as we can tweak learning rate
        self.model.compile(optimizer=self.params['tf_compile']['optimizer'],
                           loss=self.params['tf_compile']['loss'],
                           metrics=self.params['tf_compile']['metrics'])
        print(self.model.summary)
        if self.params['callback'] == 'accuracy':
            self.model.fit(self.x_train, self.y_train, epochs=10, callbacks=[AccuracyCallback])
        elif self.params['callback'] is None:
            self.model.fit(self.x_train, self.y_train, epochs=10)

    def basic_dense(self):
        self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                                 tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                                 tf.keras.layers.Dense(20, activation=tf.nn.softmax)])

        self.model.compile(optimizer=self.params['tf_compile']['optimizer'],
                           loss=self.params['tf_compile']['loss'],
                           metrics=self.params['tf_compile']['metrics'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        fit_history = self.model.fit(self.x_train, self.y_train, epochs=self.params['tf_compile']['epochs'],
                                     callbacks=[tensorboard_callback], verbose=True)
        eval_history = self.model.evaluate(self.x_test, self.y_test, verbose=True)

        filename = self.log_dir + '/train_report.txt'
        with open(filename, 'w+') as fh:
            fh.write('\n' + '#' * 20 + '\ndata statistics: \n' + '#' * 20 + '\n')
            fh.write(self.base_stats)

            fh.write('\n\n\n' + '#' * 20 + '\n' + 'model fit history: \n' + '#' * 20 + '\n')
            fh.write('params: \n')
            for k in fit_history.params.keys():
                fh.write(str(k) + ':' + str(fit_history.params[k]) + '\n')

            fh.write('\nloss, accuracy metrics: \n')
            for i in fit_history.epoch:
                fh.write('Epoch {0}/{1} - loss: {2:=0.4f} - accuracy: {3:=0.4f} \n'.format(
                    i+1, len(fit_history.epoch), fit_history.history['loss'][i], fit_history.history['accuracy'][i]))

            fh.write('\n\n\n' + '#' * 20 + '\n' + 'model summary: \n' + '#' * 20 + '\n')
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=80)

            fh.write('\n\n\n' + '#' * 20 + '\n' + 'model evaluate history: \n' + '#' * 20 + '\n')
            fh.write('loss: {0:=0.4f} - accuracy: {1:=0.4f}'.format(eval_history[0], eval_history[1]))
            fh.write('\n')

    def build_model(self):
        if self.params['mode'] == 'basic_dense':
            self.basic_dense()
        else:
            raise NotImplementedError

    def plot_samples(self, mode='training_data'):
        if mode == 'training_data':
            log_utils.plot_image_samples(self.x_train, self.y_train, random_sample_size=4, save_loc=self.log_dir)
        elif mode == 'testing_data':
            classifications = np.argmax(self.model.predict(self.x_test), axis=-1)
            log_utils.plot_pred_image_samples(self.x_test, self.y_test, classifications,
                                              random_sample_size=4, save_loc=self.log_dir)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF Build and Visualize')
    parser.add_argument('--file_info', help='file info', nargs='?', const='{}', type=str)
    parser.add_argument('--params', help='params', nargs='?', const='{}', type=str)
    args = parser.parse_args()
    file_info = eval(args.file_info)
    params = eval(args.params)

    tf_model = TF(file_info=file_info, params=params)
    tf_model.init()

    # creating the log dir in-situ ---> deprecated
    diy_log_dir = 0
    if diy_log_dir:
        tf_model.log_dir = tf_model.file_info['save_loc'] + \
                           "logs/fit/{0}_basic_dense_{1}epochs".format(tf_model.params['training_data'],
                                                                       tf_model.params['tf_compile']['epochs']) + \
                           tf_model.params['log_dir_suffix']
        if Path(tf_model.log_dir).exists() and Path(tf_model.log_dir).is_dir():
            print('log directory already exists, deleting {0}'.format(tf_model.log_dir))
            shutil.rmtree(tf_model.log_dir)
        print('creating log directory {0}'.format(tf_model.log_dir))
        os.makedirs(tf_model.log_dir)
    elif not(Path(tf_model.log_dir).exists() and Path(tf_model.log_dir).is_dir()):
        print('creating log directory {0}'.format(tf_model.log_dir))
        os.makedirs(tf_model.log_dir)
    else:
        print('directory already exists, aborting run')
        sys.exit(0)

    tf_model.plot_samples()
    tf_model.build_model()
    tf_model.plot_samples(mode='testing_data')
