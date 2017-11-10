import numpy as np 
import xgboost as xgb 
import pandas as pd
import time
from data_manager.data_preprocessor import load_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
test_path = dataset_folder_path + "test_users.csv"
output_path = "submission.csv"
n_splits = 5
n_estimators = [50, 100, 150, 200, 250]
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
min_child_weight = [1, 3, 5, 7, 9]
max_depth = [2, 4, 6, 8, 10]
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

#load
x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)

#create model
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=250, objective="multi:softprob", subsample=0.8, colsample_bytree=0.8, seed=0)

#tune max_depth and min child weight
param_grid = dict(max_depth = max_depth, min_child_weight = min_child_weight, early_stopping_rounds=25)
t = time.time()
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=0)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
print("Learning time: %.2f"%(time.time()-t))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

param_grid["max_depth"] = [grid_result.best_params_["max_depth"]]
param_grid["min_child_weight"] = [grid_result.best_params_["min_child_weight"]]

#tune subsampe and col_sample
param_grid['subsample'] = subsample
param_grid['colsample_bytree'] = colsample_bytree
t = time.time()
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=0)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
print("Learning time: %.2f"%(time.time()-t))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

param_grid['subsample'] = [grid_result.best_params_["subsample"]]
param_grid['colsample_bytree'] = [grid_result.best_params_['colsample_bytree']]

#tune learning rate and n round
param_grid['learning_rate'] = learning_rates
param_grid['n_estimators'] = n_estimators
t = time.time()
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=0)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
print("Learning time: %.2f"%(time.time()-t))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

param_grid['learning_rate'] = [grid_result.best_params_['learning_rate']]
param_grid['n_estimators'] = [grid_result.best_params_['n_estimators']]

# summarize results
print("All best parameter: ")
print(param_grid)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))