"""
Helper for saving the data in pickles and restoring the data
"""
import pickle
import os
import logging
from cryptoblipblup.config import CONFIG

PICKLE_EXTENSION = 'pickle'


def get_pickle(pickle_filename, extension=PICKLE_EXTENSION):
    """
    Loads and returns the pickle.

    :param pickle_filename: the filename without extension
    :param extension: the file extension
    :return: pickle
    """
    path = CONFIG.get('general', 'data_dir') + pickle_filename + '.' + extension
    if not os.path.isfile(path):
        return {}
    with open(path, 'rb') as f:
        data = pickle.load(f)  # load from the pickle
        logging.info('Pickle ' + pickle_filename + ' loaded')
        return data


def save_pickle(data, pickle_filename, extension=PICKLE_EXTENSION):
    """
    Serializes data and dumps it in a pickle file, not human readable

    :param pickle_filename: the filename without extension
    :param extension: the file extension
    :param data: dictionary to save
    """
    with open(CONFIG.get('general', 'data_dir') + pickle_filename + '.' + extension, 'wb') as f:
        logging.info('Saving pickle ' + pickle_filename + '...')
        pickle.dump(data, f)
        logging.info('Pickle ' + pickle_filename + ' saved')
