import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

#dataset path
base_path = os.path.dirname(os.getcwd())
train_path = base_path + "/dataset/train_session_v8.csv"
test_path = base_path + "/dataset/test_session_v8.csv"
output_path = base_path + "/dataset/submission_gscv_rf.csv"

#params for grid search
n_splits = 5
n_estimators = [500, 1000, 1200, 1500, 1800]
min_samples_leaf = [8, 9, 10, 11, 12, 13, 14, 15]
max_features = [16, 18, 20, 22, 25, 30]
param_grid = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features)

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#grid search cv
model = RandomForestClassifier(n_estimators=1000, min_samples_leaf=10, max_features=20)
t = time.time()
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv='kfold', verbose=10)
grid_result = grid_search.fit(x_train, y_train)
print("Learning time: %.2f"%(time.time()-t))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))