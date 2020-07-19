import argparse
from tree import tree, signal_function, node
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import copy
import time

start = time.time()

def load_stock_data(path, name):
    '''
    Loads saved stock data into a dataframe, typically for investigative use, should not be used in important programs
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file loaded, must not have any file denotion in it
    :return: loads the stock data into a dataframe
    '''
    return pd.read_csv(path+name+'.csv', index_col=0)

text = 'argument parser for the training software'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--dir', '-d', help='working directory of program')
parser.add_argument('--iter', '-i', help='current generation of training process')
args = parser.parse_args()

root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\Data\\'
stock_dat = root + 'fin_base\\'
project_root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\generations\\'+args.dir

curr_gen = args.iter

# load everything from previous generation
# load parameters
with open(project_root+'Training Basis\\init.params', 'rb') as f:
    init_params = pickle.load(f)

forest_size = init_params['forest_size']
half_forest = init_params['half_forest']
num_stocks_in_training_universe = init_params['universe_size']
tree_size_range = init_params['tree_size']
data_range = init_params['data_range']
num_generation = init_params['num_gens']

print('\n'+60*' '+' _____________________')
print(60*' '+'|  Generation '+str(curr_gen)+' of '+str(num_generation)+'  |')

# load universe
with open(project_root+'Training Basis\\training_stocks.unvr', 'rb') as f:
    data = [load_stock_data(path=stock_dat, name=stock)[data_range[0]: data_range[-1]]
            for stock in pickle.load(f)]

# load recent forest
with open(project_root+'Generation Data\\Forests\\gen'+str(curr_gen)+'.frst', 'rb') as f:
    prev_group = pickle.load(f)

# load statistics
with open(project_root+'Generation Data\\Statistics\\gen'+str(curr_gen)+'.stat', 'rb') as f:
    train_stats = pickle.load(f)

fit_group = train_stats['fit_group']

# mutate fit group into a new generation
forest = []
for t in fit_group[:, 0]:
    # make two copies of each model and mutate them
    #new_tree_1 = tree(rand_gen=True, size=np.random.randint(tree_size_range[0], tree_size_range[-1]))
    new_tree_1 = copy.copy(t)
    new_tree_1.mutate()
    forest.append(new_tree_1)
    # scheme 1: second tree will be another mutant of the fit group
    #new_tree_2 = copy.copy(t)
    #new_tree_2.mutate()
    # scheme 2: second tree will be randomly generated
    new_tree_2 = tree(rand_gen=True, size=np.random.randint(tree_size_range[0], tree_size_range[-1]))
    forest.append(new_tree_2)

# simulate new generation and get scores
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

# save generations
with open(project_root+'Generation Data\\Forests\\gen'+str(int(curr_gen)+1)+'.frst', 'wb') as f:
    pickle.dump(np.array(eval_forest), f)

# combine the models and scores from the fit group and the mutated group to determine the next fit group
combined_models_and_scores = np.concatenate((fit_group, eval_forest), axis=0)
scores = combined_models_and_scores[:, 1]

# determine fit group and save under generational stats
ids = np.argsort(scores)[::-1]
top_ids, bottom_ids = ids[:half_forest], ids[half_forest:]
fit_group, worst_group = combined_models_and_scores[top_ids], combined_models_and_scores[bottom_ids]

worst = np.argsort(scores)

print('Top 5:                                                                                                                                Bottom 5:')
for i in range(5):
    print(fit_group[i], '                                    ', worst_group[i])
print(fit_group[:5], '\n')

avg_runtime = train_stats['runtime']
train_stats['best_model'] = fit_group[:, 0][np.argmax(fit_group[:, 1])]
train_stats['best_score'] = np.max(fit_group[:, 1])
train_stats['average_score'] = np.mean(fit_group[:, 1])
train_stats['worst_score'] = np.min(fit_group[:, 1])
train_stats['runtime'].append(time.time()-start)
train_stats['fit_group'] = fit_group
train_stats['scores'].append(np.mean(fit_group[:, 1]))

with open(project_root+'Generation Data\\Statistics\\gen'+str(int(curr_gen)+1)+'.stat', 'wb') as f:
    pickle.dump(train_stats, f)

avggentime = np.mean(train_stats['runtime'])

timeremaining = avggentime*(int(num_generation)-int(curr_gen))
if timeremaining >= 60: # minutes
    if timeremaining >= 60*60: # hours
        if timeremaining >= 60*60*24: # days
            timeremaining /= (60*60*24)
            timeremaining = '{:.2f} d'.format(timeremaining)
        else:
            timeremaining /= (60*60)
            timeremaining = '{:.2f} h'.format(timeremaining)
    else:
        timeremaining /= 60
        timeremaining = '{:.2f} m'.format(timeremaining)
else:
    timeremaining = '{:.2f} s'.format(timeremaining)

if avggentime >= 60: # minutes
    if avggentime >= 60*60: # hours
        if avggentime >= 60*60*24: # days
            avggentime /= (60*60*24)
            avggentime = '{:.2f} d'.format(avggentime)
        else:
            avggentime /= (60*60)
            avggentime = '{:.2f} h'.format(avggentime)
    else:
        avggentime /= 60
        avggentime = '{:.2f} m'.format(avggentime)
else:
    avggentime = '{:.2f} s'.format(avggentime)

totaltime = time.time()-train_stats['start_time']
if totaltime >= 60: # minutes
    if totaltime >= 60*60: # hours
        if totaltime >= 60*60*24: # days
            totaltime /= (60*60*24)
            totaltime = '{:.2f} d'.format(totaltime)
        else:
            totaltime /= (60*60)
            totaltime = '{:.2f} h'.format(totaltime)
    else:
        totaltime /= 60
        totaltime = '{:.2f} m'.format(totaltime)
else:
    totaltime = '{:.2f} s'.format(totaltime)

fraction_complete = int(130.5*int(curr_gen)/int(num_generation))
print('Best Score {:.2f} | Average Score {:.2f} | Worst Score {:.2f} | '
      'Avg. Generation Runtime {} | Total Runtime {} | Time Remaining {}'.format(train_stats['best_score'],
                                                                     train_stats['average_score'],
                                                                     train_stats['worst_score'],
                                                                     avggentime,
                                                                     totaltime,
                                                                     timeremaining
                                                             ))
print('Progress: ['+fraction_complete*'='+'>'+(130-fraction_complete)*'-'+']\n')

if int(curr_gen) == int(num_generation):
    print('Training Complete!')
    with open(project_root+'Final Result\\final.tree', 'wb') as f:
        pickle.dump(train_stats['best_model'], f)
    plt.scatter([t for t in range(len(train_stats['runtime']))], train_stats['runtime'])
    plt.xlabel('Generation')
    plt.ylabel('Training Time')
    plt.title('Training Time per Generation')
    plt.grid()
    plt.show()