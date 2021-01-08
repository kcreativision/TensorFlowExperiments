import numpy as np
import matplotlib.pyplot as plt


def plot_image_samples(images, true_labels, random_sample_size, save_loc):
    print('#' * 100)
    print('plotting sample data from training')
    print('#' * 100)
    samples_output = dict()
    for label in np.unique(true_labels):
        label_subset = [images[i] for i in np.where(true_labels == label)][0]
        random_indices = np.random.randint(low=0, high=len(label_subset), size=random_sample_size, dtype=int)
        samples_output[label] = [label_subset[t] for t in random_indices]
        print('label: {0}, sampled: {1}'.format(label, random_sample_size))

    x_len = random_sample_size
    y_len = len(samples_output.keys())
    figs, axs = plt.subplots(nrows=y_len, ncols=x_len, figsize=(9, 6 * y_len / x_len),
                             subplot_kw={'xticks': [], 'yticks': []})

    for label_idx, label in enumerate(list(samples_output.keys())):
        plots = samples_output[label]
        for idx, plot in enumerate(plots):
            axs[label_idx, idx].imshow(plot, aspect='auto')
            if idx == 0:
                axs[label_idx, idx].set_ylabel('true label: {0}'.format(label))

    if not save_loc.endswith('/'):
        save_loc += '/'
    figs.savefig(save_loc+'training_samples.png', bbox_inches='tight')
    plt.close(figs)
    print('training data sampled and plotted to {0}'.format(save_loc+'training_samples.png'))


def plot_pred_image_samples(images, true_labels, predicted_labels, random_sample_size, save_loc):
    print('#' * 100)
    print('plotting sample data from predictions')
    print('#' * 100)
    assert len(true_labels) == len(predicted_labels), "lengths of true labels and predicted labels don't match!"
    samples_output = dict()
    pred_labels_output = dict()
    for label in np.unique(true_labels):
        tru_label_subset = [images[i] for i in np.where(true_labels == label)][0]
        pred_label_subset = [predicted_labels[i] for i in np.where(true_labels == label)][0]
        random_indices = np.random.randint(low=0, high=len(tru_label_subset), size=random_sample_size, dtype=int)
        samples_output[label] = [tru_label_subset[t] for t in random_indices]
        pred_labels_output[label] = [pred_label_subset[t] for t in random_indices]
        print('label: {0}, sampled: {1}'.format(label, random_sample_size))

    x_len = random_sample_size
    y_len = len(samples_output.keys())
    figs, axs = plt.subplots(nrows=y_len, ncols=x_len, figsize=(9, 6 * y_len / x_len),
                             subplot_kw={'xticks': [], 'yticks': []})

    for label_idx, label in enumerate(list(samples_output.keys())):
        plots = samples_output[label]
        pred_labels = pred_labels_output[label]
        for idx, plot in enumerate(plots):
            axs[label_idx, idx].imshow(plot, aspect='auto')
            if idx == 0:
                axs[label_idx, idx].set_ylabel('true label: {0}'.format(label))
                axs[label_idx, idx].set_title('predicted: {0}'.format(pred_labels[idx]))
            else:
                axs[label_idx, idx].set_title('{0}'.format(pred_labels[idx]))

    if not save_loc.endswith('/'):
        save_loc += '/'
    figs.savefig(save_loc+'predictions_samples.png', bbox_inches='tight')
    plt.close(figs)
    print('prediction data sampled and plotted to {0}'.format(save_loc + 'predictions_samples.png'))
