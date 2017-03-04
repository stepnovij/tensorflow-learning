import gzip
import logging
import os
import pickle
from collections import namedtuple

from urllib import request
from urllib.parse import urljoin

import numpy as np

LOGGING_FORMAT = '%(asctime)s - %(message)s'

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATASETS = [
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte'
]
KB_IN_MB = 1024**2
MAGIC_NUMBERS = {'label': 2049, 'data': 2051}

PATH_TO_STORE_FILES = 'data'
PICKLE_FILE = 'data_set.pkl'

IMAGE_XSIZE = 28
IMAGE_YSIZE = 28
IMAGE_SIZE = IMAGE_XSIZE, IMAGE_YSIZE

TRAIN_LABEL_PREFIX = 'train-labels'
TRAIN_DATA_PREFIX = 'train-images'
TEST_LABEL_PREFIX = 't10k-labels'
TEST_DATA_PREFIX = 't10k-images'
NUM_LABELS = 10

DataLabel = namedtuple('DataLabel', ['label', 'data'], verbose=True)


logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


class HTTPClient:
    def __init__(self, useragent=None):
        # Currently not needed:
        self.useragent = useragent

    def get(self, url, file_path):
        full_url = urljoin(url, file_path)
        logging.info('GET %s', full_url)
        response = request.urlopen(full_url)
        logging.info('Content-Length: %s', response.length/KB_IN_MB)
        return response.read()


def get_files():
    files = dict()
    if not os.path.exists(PATH_TO_STORE_FILES):
        os.makedirs(PATH_TO_STORE_FILES)
    client = HTTPClient()
    for dataset in DATASETS:
        file_path = os.path.join(PATH_TO_STORE_FILES, dataset)
        files[dataset] = file_path
        if os.path.isfile(file_path):
            logging.info('File found by path: %s', file_path)
            continue
        result = client.get(SOURCE_URL, dataset)
        with gzip.open(os.path.join(file_path), 'wb') as f:
            f.write(result)
    return files


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(8), dtype=dt)


def read_data(bytestream, path):
    magic, _ = _read32(bytestream)
    dt8 = np.dtype(np.uint8)
    dt32 = np.dtype(np.uint32).newbyteorder('>')
    assert magic in MAGIC_NUMBERS.values(), 'Invalid number {} file:{}'.format(magic, path)
    if magic == MAGIC_NUMBERS['label']:
        return np.frombuffer(bytestream.read(), dtype=dt8)
    else:
        _ = np.frombuffer(bytestream.read(8), dtype=dt32)
        return np.frombuffer(bytestream.read(), dtype=dt8)


def extract_images():
    result = {}
    for label, path in get_files().items():
        with gzip.open(path, 'rb') as f:
            result[label] = read_data(f, path)
    return result


def get_dataset():
    dataset = extract_images()
    with open(os.path.join(PATH_TO_STORE_FILES, PICKLE_FILE), 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


def load_dataset():
    logging.info('Loading dataset')
    pickled_file_path = os.path.join(PATH_TO_STORE_FILES, PICKLE_FILE)
    if os.path.isfile(pickled_file_path):
        logging.info('Pickled file found by path: %s', pickled_file_path)
        with open(os.path.join(PATH_TO_STORE_FILES, PICKLE_FILE), 'rb') as f:
            return pickle.load(f)
    return get_dataset()


def get_data_in_proportion(data, labels, idx):
    return labels[idx], data[idx]

def split_train_ds(data, labels, split_proportion):
    size = labels.shape
    train_idx = np.random.choice([True, False], size=(size[0],), p=[split_proportion,
                                                                1 - split_proportion])
    validation_idx = np.random.choice([True, False], size=(size[0],), p=[1 - split_proportion,
                                                                split_proportion])
    data.shape = (-1, IMAGE_XSIZE, IMAGE_YSIZE)

    train_label, train_data = get_data_in_proportion(data, labels, train_idx)
    validation_label, validation_data = get_data_in_proportion(data, labels, validation_idx)

    train_dataset = DataLabel(label=train_label, data=train_data)
    validation_dataset = DataLabel(label=validation_label, data=validation_data)

    train_dataset.data.shape = (-1, IMAGE_XSIZE*IMAGE_YSIZE)
    validation_dataset.data.shape = (-1, IMAGE_XSIZE*IMAGE_YSIZE)

    return train_dataset, validation_dataset


def get_data_and_label(data_dict, label_prefix, data_prefix):
    data, label = None, None
    for key, value in data_dict.items():
        if data_prefix in key:
            data = value
        elif label_prefix in key:
            label = value
    return data, label


def load_test_train_validation_ds(split_proportion=0.6):
    dataset = load_dataset()

    train_data, train_label = get_data_and_label(dataset, TRAIN_LABEL_PREFIX, TRAIN_DATA_PREFIX)
    train_dataset, validation_dataset = split_train_ds(train_data, train_label, split_proportion)

    test_data, test_label = get_data_and_label(dataset, TEST_LABEL_PREFIX, TEST_DATA_PREFIX)
    test_data.shape = (-1, IMAGE_XSIZE*IMAGE_YSIZE)
    test_dataset = DataLabel(label=test_label, data=test_data)

    return test_dataset, train_dataset, validation_dataset