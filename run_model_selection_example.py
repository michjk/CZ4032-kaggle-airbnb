import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import load_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2_NDF_vs_non_NDF.csv"
test_path = dataset_folder_path + "test_users.csv"
output_path = "submission.csv"
base_path = os.path.dirname(__file__)
dataset_folder_path = "dataset/"
train_path = base_path + dataset_folder_path + "train_users_2.csv"
test_path = base_path + dataset_folder_path + "test_users.csv"
output_path = base_path + "submission.csv"
n_splits = 5
n_estimators = [100, 200, 500, 800, 1000]
min_samples_leaf = [3, 4, 5, 6, 7, 8, 9, 10]
max_features = [12, 13, 14, 15, 16, 20, 50, 70, 90, 110, 130, 161]
param_grid = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features)

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)

#train
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=12)

t = time.time()
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold, verbose=10)
grid_result = grid_search.fit(x_train, y_train)
print("Learning time: %.2f"%(time.time()-t))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))