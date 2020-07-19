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

root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\Data\\'
stock_dat = root + 'fin_base\\'
project_root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\generations\\'


def load_stock_data(path, name):
    '''
    Loads saved stock data into a dataframe, typically for investigative use, should not be used in important programs
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file loaded, must not have any file denotion in it
    :return: loads the stock data into a dataframe
    '''
    return pd.read_csv(path+name+'.csv', index_col=0)


forest_size = int(args.forestsize)
half_forest = int(forest_size / 2)
num_stocks_in_training_universe = int(args.universesize)
tree_size_range = (int(args.treelower), int(args.treeupper))
data_range = (-300, -1)
num_generations = int(args.numgen)

init_params = {'forest_size': forest_size,
               'half_forest': half_forest,
               'universe_size': num_stocks_in_training_universe,
               'tree_size': tree_size_range,
               'data_range': data_range,
               'num_gens': num_generations}

# hot start argument
if args.hot_start is not None:  # hot start protocol
    hot_start_path = project_root + args.hot_start + '\\Generation Data\\Statistics\\'
    most_recent = max([int(f.split(sep='gen')[-1].split(sep='.stat')[0]) for f in listdir(hot_start_path)if isfile(join(hot_start_path, f))])
    with open(hot_start_path + 'gen' + str(most_recent) + '.stat', 'rb') as f:
        orig_train_stats = pickle.load(f)
    forest = []
    for t in orig_train_stats['fit_group'][:, 0]:
        new_model_1 = copy.copy(t)
        forest.append(new_model_1)
        new_model_2 = copy.copy(t)
        new_model_2.mutate()
        forest.append(new_model_2)
    forest_size = len(forest)
    half_forest = int(forest_size / 2)
    init_params['forest_size'] = forest_size
    init_params['half_forest'] = half_forest
else:  # normal start
    # randomly generate the first forest
    forest = np.array([tree(rand_gen=True, size=np.random.randint(tree_size_range[0], tree_size_range[-1]))
                       for i in range(forest_size)])

project_root = project_root + args.dir



# randomly select stocks from the database and make the universe
'''n_random_stocks = ['AAPL',
          'MSFT',
          'AMZN',
          'GOOG',
          'FB',
          'BRKS',
          'BRKR',
          'BRKL',
          'V',
          'JNJ',
          'WMT',
          'MA',
          'PG',
          'UNH',
          'JPM',
          'HD',
          'INTC',
          'NVDA',
          'VZ',
          'TSLA',
          'T',
          'ADBE',
          'NFLX',
          'PYPL',
          'DIS',
          'BAC',
          'MRK',
          'KO',
          'CSCO',
          'PFE',
          'XOM',
          'PEP',
          'CMCSA',
          'ABBV',
          'CRM',
          'ORCL',
          'CVX',
          'ABT',
          'LLY',
          'NKE',
          'AMGN',
          'TMO',
          'ACN',
          'MCD',
          'COST',
          'BMY',
          'DHR',
          'AVGO',
          'MDT',
          'NEE',
          'AMT',
          'LIN']'''
#n_random_stocks = ['^GSPC']
stocks = [f.split(sep='.')[0] for f in listdir(stock_dat) if isfile(join(stock_dat, f))]
n_random_stocks = np.random.permutation(stocks)[:num_stocks_in_training_universe]
data = [load_stock_data(path=stock_dat, name=stock)[data_range[0]: data_range[-1]]
        for stock in n_random_stocks]
print('Stocks in Universe:\n', n_random_stocks)

# save the stocks and parameters
with open(project_root+'Training Basis\\training_stocks.unvr', 'wb') as f:
    pickle.dump(n_random_stocks, f)
with open(project_root+'Training Basis\\init.params', 'wb') as f:
    pickle.dump(init_params, f)

print('Evaluating first generation of models. . .')
# evaluate forest and attach scores
scores = []
for df in data:
    scores.append([t.basic_simulation(df, freq_weighted_scoring=True) for t in forest])
scores = np.mean(np.array(scores), axis=0)
nans = np.isnan(scores)
scores[nans] = 0

eval_forest = []
for t, s in zip(forest, scores):
    eval_forest.append([t, s])
eval_forest = np.array(eval_forest)

# save first forest
with open(project_root+'Generation Data\\Forests\\gen1.frst', 'wb') as f:
    pickle.dump(eval_forest, f)

ids = np.argsort(scores)[::-1][:half_forest]
fit_group = eval_forest[ids]

train_stats = {'best_model': forest[np.argmax(scores)],
               'best_score': np.max(scores),
               'average_score': np.mean(scores),
               'worst_score': np.min(scores),
               'runtime': [time.time()-start],
               'start_time': start,
               'fit_group': fit_group,
               'scores': [fit_group[:, 1]]}

with open(project_root+'Generation Data\\Statistics\\gen1.stat', 'wb') as f:
    pickle.dump(train_stats, f)

print('Initialization done!')