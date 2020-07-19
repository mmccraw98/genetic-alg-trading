import numpy as np
import pandas as pd
from tree import tree, signal_function, node
import pickle
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join


def save_obj_to_path(obj, path, name):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path + name, 'rb') as f:
        pickle.dump(obj, f)


def get_folders(path):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


def get_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\generations\\'
training_files = [f.split(sep='.')[0] for f in listdir(root) if isfile(join(root, f))]

# for folder in get_folders(root):
folder = get_folders(root)[1]
training_folder = get_folders(folder)[1]
generations = get_files(get_folders(folder)[1])
ranked_index = np.argsort([float(f.split(sep='gen')[-1].split(sep='frst')[0]) for f in generations])
files_to_open = [training_folder + '\\' + gen_name for gen_name in np.array(generations)[ranked_index]]

models, scores = [], []
for i, file in enumerate(files_to_open):
    with open(file, 'rb') as f:
        gen_data = pickle.load(f)

    models.append(gen_data[:, 0])
    scores.append(gen_data[:, 1])

plt.plot(scores)
plt.show()

print([n.get_attributes() for n in models[-1][-1].node_list])