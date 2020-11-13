import tensorflow as tf
import numpy as np
import argparse


class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # TODO add parameter for accuracy threshold
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


class BuildTF(object):
    def __init__(self, file_info, params):
        self.file_info = file_info
        self.params = params

        self.assert_fileinfo()
        self.assert_params()

        self.x_train, self.y_train = np.array, np.array
        self.x_test, self.y_test = np.array, np.array
        self.read_data()
        self.preprocess_data()
        # self.data_stats()

    def assert_fileinfo(self):
        assert 'training_data' in self.file_info.keys(), 'training_data in file_info is compulsory'

    def assert_params(self):
        if 'normalise_method' not in self.params.keys():
            self.params['normalise_method'] = None
        if 'callback' not in self.params.keys():
            self.params['callback'] = None
        assert 'tf_compile' in self.params.keys(), "compile params are compulsory"

    def read_data(self):
        if self.file_info['training_data'] == 'fashion_mnist':
            print('reading fashion_mnist', end=' ', flush=True)
            fmnist = tf.keras.datasets.fashion_mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fmnist.load_data()
            print('x_train shape: {0}'.format(np.shape(self.x_train)), end=' ', flush=True)
            print('x_test shape: {0}'.format(np.shape(self.x_train)))
        else:
            raise NotImplementedError

    def preprocess_data(self):
        if self.params['normalise_method'] == 'pixel':
            print('normalising by 255.0')
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        elif self.params['normalise_method'] is None:
            print('not normalising the input data, neural nets might not converge')
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
        # TODO check binayr_cross_entropy for binary classification
        # TODO check RMSProp as optimizer as we can tweak learning rate
        self.model.compile(optimizer=self.params['tf_compile']['optimizer'],
                      loss=self.params['tf_compile']['loss'],
                      metrics=self.params['tf_compile']['metrics'])
        print(self.model.summary)
        if self.params['callback'] == 'accuracy':
            self.model.fit(self.x_train , self.y_train, epochs=10, callbacks=[AccuracyCallback])
        elif self.params['callback'] is None:
            self.model.fit(self.x_train, self.y_train, epochs=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF build')
    parser.add_argument('--file_info', help='file info')
    parser.add_argument('--params', help='params')
    args = parser.parse_args()
    # TODO add default args
    parent_file_info = eval(args.file_info)
    parent_params = eval(args.params)
    print('file_info', parent_file_info)
    print('params')
    for k in parent_params.keys():
        print(k, type(parent_params[k]),parent_params[k])

    tf_model = BuildTF(parent_file_info, parent_params)
    tf_model.build()
