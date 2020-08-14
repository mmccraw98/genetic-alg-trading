import pickle
import argparse
import numpy as np
import pandas as pd
from tree import tree, signal_function, node
from supporting_functions import get_recent, get_time_report, get_stats_report
from copy import copy
import time

start = time.time()

text = 'argument parser for the initialization of the training software'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--dir', '-d', help='working directory of program')
args = parser.parse_args()

project_root = 'C:\\Users\\PC\\PycharmProjects\\SelfGenAlg\\Data\\'
model_root = project_root + 'Training\\'
data_root = project_root + 'Stocks\\'
dir = model_root + args.dir

with open(dir + '\\params.pkl', 'rb') as f:
    params_dict = pickle.load(f)

with open(dir + '\\stats.pkl', 'rb') as f:
    stats_dict = pickle.load(f)

print('\n'+60*' '+' _____________________')
print(60*' '+'|  Generation '+str(params_dict['current_gen'] + 1)+' of '+str(params_dict['numgen'])+'  |')


if params_dict['current_gen'] == 0:
    fit_forest, fit_scores = [], []
    # generate first group of models
    if params_dict['hot_start']:
        # use hot_start
        hot_start_root = model_root + params_dict['hot_start']
        with open(get_recent(path=hot_start_root + '\\Generations\\'), 'rb') as f:
            new_forest, new_scores = pickle.load(f)
    else:
        # generate randomly
        new_forest = [tree(rand_gen=True, size=np.random.randint(
                      params_dict['treelower'], params_dict['treeupper'])) for i in range(params_dict['forestsize'])]
else:
    # load previous generation's models, rank, and mutate
    # load
    with open(get_recent(path=dir + '\\Generations\\'), 'rb') as f:
        fit_forest, fit_scores = pickle.load(f)
    # rank, mutate
    if params_dict['mutate_scheme'] == 1:
        # mutate full group from top half
        top_half = np.argsort(fit_scores)[::-1][:int(params_dict['universesize'] / 2)]
        new_forest = []
        for model in np.array(fit_forest)[top_half]:
            new_model_1, new_model_2 = copy(model), copy(model)
            new_model_1.mutate()
            new_forest.append(new_model_1)
            new_model_2.mutate()
            new_forest.append(new_model_2)
    elif params_dict['mutate_scheme'] == 2:
        # muatate half, randomly generate half
        top_half = np.argsort(fit_scores)[::-1][:int(params_dict['universesize'] / 2)]
        new_forest = []
        for model in np.array(fit_forest)[top_half]:
            new_model_1 = copy(model)
            new_model_1.mutate()
            new_forest.append(new_model_1)
            new_model_2 = tree(rand_gen=True, size=np.random.randint(params_dict['treelower'], params_dict['treeupper']))
            new_forest.append(new_model_2)
    elif params_dict['mutate_scheme'] == 3:
        # mutate top quarter, randomly generate three quarters
        top_quarter = np.argsort(fit_scores)[::-1][:int(params_dict['universesize'] / 4)]
        new_forest = []
        for model in np.array(fit_forest)[top_quarter]:
            new_model = copy(model)
            new_model.mutate()
            new_forest.append(new_model)
        while len(new_forest) != params_dict['forestsize']:
            new_forest.append(tree(rand_gen=True, size=np.random.randint(params_dict['treelower'], params_dict['treeupper'])))

# load stock data
stocks = [pd.read_csv(data_root + 'All\\' + stock, index_col=0) for stock in params_dict['universe']]

# evaluate models
new_scores = []
for df in stocks:
    new_scores.append([model.basic_simulation(df, freq_weighted_scoring=True) for model in new_forest])
new_scores = np.mean(np.array(new_scores), axis=0)
nans = np.isnan(new_scores)
new_scores[nans] = 0

# combine scores, rank best models from new_forest and fit_forest
total_scores, total_forest = np.concatenate((new_scores, fit_scores)), np.concatenate((new_forest, fit_forest))
best_scores = np.argsort(total_scores)[::-1][:params_dict['forestsize']]
best_models = np.array(total_forest)[best_scores]

# save
with open(dir + '\\Generations\\Gen_' + str(params_dict['current_gen']), 'wb') as f:
    pickle.dump((best_models, best_scores), f)

stats_dict['best_score_from_gen'].append(max(new_scores))
if params_dict['current_gen'] == 0:
    stats_dict['best_score_total'].append(max(best_scores))
else:
    if max(best_scores) > stats_dict['best_score_total'][-1]:
        stats_dict['best_score_total'].append(max(best_scores))
    else:
        stats_dict['best_score_total'].append(stats_dict['best_score_total'][-1])
stats_dict['worst_score_from_gen'].append(min(new_scores))
stats_dict['gen_run_time'].append(time.time() - start)
stats_dict['avg_size_from_gen'].append(np.mean([model.get_size() for model in new_forest]))
stats_dict['avg_size_total'].append(np.mean([model.get_size() for model in best_models]))
stats_dict['avg_best_score_total'].append(np.mean(best_scores))

with open(dir + '\\stats.pkl', 'wb') as f:
    pickle.dump(stats_dict, f)

params_dict['current_gen'] += 1

with open(dir + '\\params.pkl', 'wb') as f:
    pickle.dump(params_dict, f)

get_stats_report(stats_dict=stats_dict)
get_time_report(stats_dict=stats_dict, params_dict=params_dict)

fraction_complete = int(130.5*params_dict['current_gen']/params_dict['numgen'])
print('Progress: ['+fraction_complete*'='+'>'+(130-fraction_complete)*'-'+']\n')
