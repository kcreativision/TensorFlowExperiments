import numpy as np
import matplotlib.pyplot as plt


def plot_image_samples(images, true_labels, random_sample_size, save_loc):
    print('#' * 100)
    print('plotting sample data from training')
    print('#' * 100)
    samples_output = dict()
    for label in np.unique(true_labels):
        label_subset = [images[i] for i in np.where(true_labels == label)][0]
        rng = np.random.default_rng()
        random_indices = rng.choice(a=len(label_subset), size=random_sample_size, replace=False)
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

    plt.suptitle('Sample Images - Training Data')

    if not save_loc.endswith('/'):
        save_loc += '/'
    figs.savefig(save_loc+'training_samples.png', bbox_inches='tight')
    plt.close(figs)
    print('training data sampled and plotted to {0}'.format(save_loc+'training_samples.png'))


def plot_pred_image_samples(images, true_labels, predicted_labels, random_sample_size, save_loc):
    print('#' * 100)
    print('plotting sample data from correct predictions')
    print('#' * 100)
    assert len(true_labels) == len(predicted_labels), "lengths of true labels and predicted labels don't match!"
    samples_correct_output = dict()
    pred_labels_correct_output = dict()
    for label in np.unique(true_labels):
        tru_label_subset_correct = [images[i] for i in np.where((true_labels == label) &
                                                                (true_labels == predicted_labels))][0]
        pred_label_subset_correct = [predicted_labels[i] for i in np.where((true_labels == label) &
                                                                           (true_labels == predicted_labels))][0]
        rng = np.random.default_rng()
        if len(tru_label_subset_correct) >= random_sample_size:
            random_indices = rng.choice(a=len(tru_label_subset_correct), size=random_sample_size, replace=False)
        else:
            random_indices = rng.choice(a=len(tru_label_subset_correct), size=len(tru_label_subset_correct),
                                        replace=False)
        samples_correct_output[label] = [tru_label_subset_correct[t] for t in random_indices]
        pred_labels_correct_output[label] = [pred_label_subset_correct[t] for t in random_indices]
        print('label: {0}, sampled: {1}'.format(label, random_sample_size))

    x_len = random_sample_size
    y_len = len(samples_correct_output.keys())
    figs, axs = plt.subplots(nrows=y_len, ncols=x_len, figsize=(9, 6 * y_len / x_len),
                             subplot_kw={'xticks': [], 'yticks': []})

    for label_idx, label in enumerate(list(samples_correct_output.keys())):
        plots = samples_correct_output[label]
        pred_labels = pred_labels_correct_output[label]
        for idx, plot in enumerate(plots):
            axs[label_idx, idx].imshow(plot, aspect='auto')
            if idx == 0:
                axs[label_idx, idx].set_ylabel('true label: {0}'.format(label))
                axs[label_idx, idx].set_title('predicted: {0}'.format(pred_labels[idx]))
            else:
                axs[label_idx, idx].set_title('{0}'.format(pred_labels[idx]))

    plt.suptitle('Sample Images - Correct Predictions')
    if not save_loc.endswith('/'):
        save_loc += '/'
    figs.savefig(save_loc+'correct_predictions_samples.png', bbox_inches='tight')
    plt.close(figs)
    print('prediction data sampled and plotted to {0}'.format(save_loc + 'correct_predictions_samples.png'))

    print('#' * 100)
    print('plotting sample data from incorrect predictions')
    print('#' * 100)

    samples_incorrect_output = dict()
    pred_labels_incorrect_output = dict()
    for label in np.unique(true_labels):
        tru_label_subset_incorrect = [images[i] for i in np.where((true_labels == label) &
                                                                  (true_labels != predicted_labels))][0]
        pred_label_subset_incorrect = [predicted_labels[i] for i in np.where((true_labels == label) &
                                                                             (true_labels != predicted_labels))][0]
        rng = np.random.default_rng()
        if len(tru_label_subset_incorrect) >= random_sample_size:
            random_indices = rng.choice(a=len(tru_label_subset_incorrect), size=random_sample_size, replace=False)
        else:
            random_indices = rng.choice(a=len(tru_label_subset_incorrect), size=len(tru_label_subset_incorrect),
                                        replace=False)
        samples_incorrect_output[label] = [tru_label_subset_incorrect[t] for t in random_indices]
        pred_labels_incorrect_output[label] = [pred_label_subset_incorrect[t] for t in random_indices]
        print('label: {0}, sampled: {1}'.format(label, random_sample_size))

    x_len = random_sample_size
    y_len = len(samples_incorrect_output.keys())
    figs, axs = plt.subplots(nrows=y_len, ncols=x_len, figsize=(9, 6 * y_len / x_len),
                             subplot_kw={'xticks': [], 'yticks': []})

    for label_idx, label in enumerate(list(samples_incorrect_output.keys())):
        plots = samples_incorrect_output[label]
        pred_labels = pred_labels_incorrect_output[label]
        for idx, plot in enumerate(plots):
            axs[label_idx, idx].imshow(plot, aspect='auto')
            if idx == 0:
                axs[label_idx, idx].set_ylabel('true label: {0}'.format(label))
                axs[label_idx, idx].set_title('predicted: {0}'.format(pred_labels[idx]))
            else:
                axs[label_idx, idx].set_title('{0}'.format(pred_labels[idx]))
    plt.suptitle('Sample Images - Incorrect Predictions')
    if not save_loc.endswith('/'):
        save_loc += '/'
    figs.savefig(save_loc+'incorrect_predictions_samples.png', bbox_inches='tight')
    plt.close(figs)
    print('prediction data sampled and plotted to {0}'.format(save_loc + 'incorrect_predictions_samples.png'))