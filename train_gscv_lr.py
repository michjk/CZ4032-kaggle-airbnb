import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.linear_model import LogisticRegression
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
c = [50, 70, 100, 120, 150, 180, 200]
solver = ['newton-cg', 'sag', 'lbfgs']
max_iter = [100, 150, 200, 250, 300, 500]
multi_class = ['multinomial']
param_grid = dict(C=c, solver=solver, max_iter=max_iter, multi_class=multi_class)

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#grid search cv
model = LogisticRegression(penalty='l2', C=100, solver='newton-cg', max_iter=100, multi_class='multinomial', random_state=0, n_jobs=-1)
t = time.time()
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kfold, verbose=10)
grid_result = grid_search.fit(x_train, y_train)
print("Learning time: %.2f"%(time.time()-t))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))