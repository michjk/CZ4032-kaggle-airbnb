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
train_path = dataset_folder_path + "train_users_exclude_NDF.csv"
test_path = dataset_folder_path + "test_users.csv"
output_path = "submission.csv"
n_splits = 5
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rates)

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)

#train
model = xgb.XGBClassifier(max_depth=8, n_estimators=200, objective="multi:softprob", subsample=0.5, colsample_bytree=0.5, seed=0)

t = time.time()
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=0)
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