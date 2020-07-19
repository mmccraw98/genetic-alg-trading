import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import errno


def save_obj_to_path(obj, path, name):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path + name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj_from_path(path, name):
    try:
        with open(path + name, 'rb') as f:
            return pickle.load(f)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def get_folders(path):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


def get_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def make_folders(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


def load_stock_data(path, name):
    '''
    Loads saved stock data into a dataframe, typically for investigative use, should not be used in important programs
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file loaded, must not have any file denotion in it
    :return: loads the stock data into a dataframe
    '''
    return pd.read_csv(path+name+'.csv', index_col=0)


def make_training_dir(path, dirname):
    root = path + '\\' + dirname + '\\'
    make_folders(root)
    make_folders(root + 'Universe\\')
    make_folders(root + 'Generations\\')
