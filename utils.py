import numpy as np

LOGGING_FORMAT = '%(asctime)s - %(message)s'


def accuracy(predictions, labels):
    return 100*np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/predictions.shape[0]


def convert_from_one_dim_labels(labels, num_labels):
    train_dataset = np.zeros([labels.shape[0], num_labels])
    train_dataset[np.arange(labels.shape[0]), labels] = 1
    return train_dataset
