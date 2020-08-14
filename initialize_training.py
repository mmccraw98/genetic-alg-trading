import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import copy
import time
import sys
import pickle
from tree import tree, signal_function, node
import supporting_functions as sf

start = time.time()

print('Working. . .')

text = 'argument parser for the initialization of the training software'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--specify', '-spec', help='whether or not the universe will be generated randomly or be previously specified inside this file')
parser.add_argument('--forestsize', '-fsize', help='defines the size of the forest for each generation')
parser.add_argument('--universesize', '-usize', help='defines the number of stocks in the current universe')
parser.add_argument('--treelower', '-tlb', help='defines the lower bound of tree model size')
parser.add_argument('--treeupper', '-tub', help='defines the upper bound of tree model size')
parser.add_argument('--numgen', '-ng', help='defines the number of generations for training')
parser.add_argument('--dir', '-d', help='working directory of program')
parser.add_argument('--hot_start', '-hs', help='defines whether or not to start with a pre-existing group of models')
args = parser.parse_args()

#@TODO NEED TO MOVE THE DATABSE INTO THE DATA FOLDER UNDER A FOLDER CALLED \DATABASE
#@TODO THEN PUT ALL SUB-DATABASES INTO FOLDERS WITHIN THE DATA FOLDER WITH DESCRIPTIVE NAMES
#root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\Data\\'
#stock_database = root + 'fin_base\\'

root = 'C:\\Users\\PC\\PycharmProjects\\SelfGenAlg\\Data\\'
training_dir = root + 'Training\\'
stock_database = root + 'Stocks\\'

# create parameters from user input
params = {'forest_size': int(args.forestsize),
          'half_forest': int(int(args.forestsize) / 2),
          'universe_size': int(args.universesize),
          'uni_specified': args.specify,  # need to add this to the setup - menu
          'tree_size': (int(args.treelower), int(args.treeupper)),
          'data_range': (-300, -1),  # need to add this to the setup - menu
          'num_generations': int(args.numgen)}

sf.make_training_dir(path=training_dir, dirname=args.dir)

project_root = training_dir + args.dir

# IN THE CASE THAT THE UNIVERSE SIZE IS GREATER THAN THE AVAILABLE ASSETS, DEFAULT TO LARGEST SIZE OF AVAILABLE
print(params['uni_specified'], type(params['uni_specified']))
# get stocks
if params['uni_specified'] == 'None':  # random case
    print('random')
    all_stocks = [f.split(sep='.')[0] for f in sf.get_files(stock_database + 'All\\')]
    print(all_stocks, 20*'\n')
    if params['universe_size'] > len(all_stocks):
        params['universe_size'] = len(all_stocks)
    random_stocks = np.random.permutation(all_stocks)[:params['universe_size']]
    data = [sf.load_stock_data(path=stock_database + 'All\\', name=stock)[params['data_range'][0]: params['data_range'][-1]]
            for stock in random_stocks]
else:  # specified case
    print('specified')
    all_stocks = [f.split(sep='.')[0] for f in sf.get_files(stock_database + params['uni_specified'] + '\\')]
    print(all_stocks)

#print('Stocks in Universe:\n', n_random_stocks)


# hot start



