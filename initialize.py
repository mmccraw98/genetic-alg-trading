from os.path import exists
from supporting_functions import make_training_dir, get_files
import pickle
import argparse
import numpy as np
import time

start = time.time()

text = 'argument parser for the initialization of the training software'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--dir', '-d', help='working directory of program')
args = parser.parse_args()


def numeric_prompt(prompt):
    print(prompt)
    input_val = input('> ')
    try:
        return int(input_val)
    except:
        print('{} is the incorrect type.  Please enter an int not a {}.'.format(input_val, type(input_val)))
        return numeric_prompt(prompt=prompt)


def file_prompt(prompt, root, create_new):
    print(prompt)
    input_val = input('> ')
    if create_new:
        if exists(root+input_val):
            print('{} already exists.  Please enter a different file name.'.format(input_val))
            return file_prompt(prompt=prompt, root=root, create_new=create_new)
        else:
            return input_val
    else:
        if exists(root+input_val):
            return input_val
        else:
            print('{} does not exist.  Please enter an existing file.'.format(input_val))
            return file_prompt(prompt=prompt, root=root, create_new=create_new)


def y_n_prompt(prompt):
    print(prompt)
    input_val = input('> y/n:')
    if input_val.lower() in ['y', 'n']:
        if input_val.lower() == 'y':
            return True
        else:
            return False
    else:
        print('{} not understood.  Please enter y or n.'.format(input_val))
        return y_n_prompt(prompt=prompt)


project_root = 'C:\\Users\\PC\\PycharmProjects\\SelfGenAlg\\Data\\'
model_root = project_root + 'Training\\'
data_root = project_root + 'Stocks\\'

dir = args.dir

if y_n_prompt(prompt='Use hot start protocol?'):
    hot_start = file_prompt(prompt='Please enter the name of the model set to be used for hot start.', root=model_root, create_new=False)
    print('Using {} for hot start protocol.'.format(hot_start))
else:
    hot_start = False

if y_n_prompt(prompt='Use pre-defined stock universe?'):
    specify = file_prompt(prompt='Please enter the name of the stock universe to be used for training.', root=data_root, create_new=False)
    print('Using {} for training.'.format(specify))
else:
    specify = False

if y_n_prompt(prompt='Use default parameters?'):
    forestsize = 40
    universesize = 20
    treelower = 3
    treeupper = 50
    numgen = 25
    mutate_scheme = 1
else:
    forestsize = numeric_prompt(prompt='Please enter the desired forest size.')
    universesize = numeric_prompt(prompt='Please enter the desired universe size.')
    treelower = numeric_prompt(prompt='Please enter the desired lower range for model size.')
    treeupper = numeric_prompt(prompt='Please enter the desired upper range for the model size.')
    numgen = numeric_prompt(prompt='Please enter the desired number of generations of training.')
    mutate_scheme = numeric_prompt(prompt='Please enter the desired mutation scheme (1, 2, 3).')

print('A forest of size {} containing trees of sizes between {} and {} will be trained on {} stocks for {} generations.'.format(
    forestsize, treelower, treeupper, universesize, numgen
))
print('The results will be saved under {}.'.format(dir))
if hot_start:
    print('Hot start protocol will be used, taking models from {}.'.format(hot_start))
if specify:
    print('The stocks will be pre-defined and taken from {}.'.format(specify))
    files = get_files(data_root + specify + '\\')
    if universesize > len(files):
        print('Too many stocks chosen, cutting down from {} to {}.'.format(universesize, len(files)))
        universesize = len(files)
    universe = np.random.permutation(files)[:universesize]
else:
    print('The stocks will be chosen at random.')
    files = get_files(data_root + 'All\\')
    if universesize > len(files):
        print('Too many stocks chosen, cutting down from {} to {}.'.format(universesize, len(files)))
        universesize = len(files)
    universe = np.random.permutation(files)[:universesize]

make_training_dir(path=model_root, dirname=dir)

params = {'forestsize': forestsize,
          'universesize': universesize,
          'treelower': treelower,
          'treeupper': treeupper,
          'numgen': numgen,
          'dir': dir,
          'hot_start': hot_start,
          'specify': specify,
          'current_gen': 0,
          'mutate_scheme': mutate_scheme,
          'universe': universe}

with open(model_root + dir + '\\params.pkl', 'wb') as f:
    pickle.dump(params, f)

stats_dict = {'best_score_from_gen': [],
              'best_score_total': [],
              'worst_score_from_gen': [],
              'avg_best_score_total': [],
              'gen_run_time': [time.time() - start],
              'avg_size_from_gen': [],
              'avg_size_total': [],
              'start_time': start}

with open(model_root + dir + '\\stats.pkl', 'wb') as f:
    pickle.dump(stats_dict, f)
