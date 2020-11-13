import tensorflow as tf
import numpy as np
import argparse
from tensorflow_exps.build import BuildTF


class VisualizeTF(BuildTF):
    """
    Visualizations on
        1. Convolution layers
        2. Pooling layers
    """
    def __init__(self, model, file_info, params):
        self.model = model
        self.file_info = file_info
        self.params = params

        self.assert_fileinfo()
        self.assert_params()

        self.x_test, self.y_test = np.array, np.array
        self.read_data()
        self.preprocess_data()

    def assert_fileinfo(self):
        assert 'testing_data' in self.file_info.keys(), 'training_data in file_info is compulsory'

    def load_model(self):
        layer_outputs = [layer.output for layer in self.model.layers]
        self.activation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_outputs)

    def conv_graphs(self):
        # TODO add a random test sample based conv graph
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF Build and Visualize')
    # TODO add default args
    parser.add_argument('--file_info', help='file info')
    parser.add_argument('--params', help='params')
    args = parser.parse_args()

    parent_file_info = eval(args.file_info)
    parent_params = eval(args.params)
    print('file_info', parent_file_info)
    print('params')
    for k in parent_params.keys():
        print(k, type(parent_params[k]),parent_params[k])

    tf_model = BuildTF(file_info=parent_file_info, params=parent_params)
    tf_model.build()

    tf_graphs = VisualizeTF(model=tf_model.model, file_info=parent_file_info, params=parent_params)
    tf_graphs.conv_graphs()
