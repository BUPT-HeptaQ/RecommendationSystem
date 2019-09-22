# use Jaccard similarity: number of intersection elements/ number of union elements
# here could use any recommendation algorithms

import os
import csv
from surprise import SVD, GridSearch
from surprise import Dataset
from surprise import evaluate, print_perf

# load your own data sets, assign the file path
file_path = os.path.expanduser(' ')
# tell the text reader what is the text format
reader = csv.reader(line_format='user item rating timestamp', sep=',')
# load data
data = Dataset.load_from_file(file_path, reader=reader)
# k-fold cross validation (k=5)
data.split(n_folds=5)
# try SVD matrix decomposition
algo = SVD()
# use data set to test outcomes
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# outcomes
print_perf(perf)

# adjust the parameters of algorithm
# define the parameters grid to be optimized
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
# use grid search to cross validation
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
# find the best parameter group on data set and output the best RMSE
print(grid_search.best_score['RMSE'])
# output the best parameters with RMSE
print(grid_search.best_params['RMSE'])
# output the best FCP score
print(grid_search.best_score['FCP'])
# output the parameters with FCP
print(grid_search.best_params['FCP'])

