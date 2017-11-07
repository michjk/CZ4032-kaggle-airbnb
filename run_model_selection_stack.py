import numpy as np 
import xgboost as xgb 
import pandas as pd
import time

from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

#dataset path
dataset_folder_path = "../dataset/"
train_path_1 = dataset_folder_path + "train_users_2_NDF_vs_non_NDF.csv"
train_path_2 = dataset_folder_path + "train_users_2.csv"
test_path = dataset_folder_path + "test_users.csv"
output_path = "submission.csv"
nfold = 5

x_train_1, y_train_1, x_test, id_test, label_encoder_1 = load_preprocessed_data(train_path_1, test_path)

''' 
params = {}
params['objective'] = "binary:logistic"
params['num_class'] = 2
params['max_depth'] = 5
params['min_child_weight'] = 10
params['eta'] = 0.2
params['subsample'] = 0.5
params['colsample_bytree'] = 0.5
params['seed'] = 0
train_dmat = xgb.DMatrix(x_train_1, y_train_1)

model_1 = xgb.train(params, train_dmat, num_boost_round=150, early_stopping_rounds=15)
#model_1 = xgb.cv(params, train_dmat, num_boost_round=15, early_stopping_rounds=15, metrics=['logloss'], nfold=cv, seed=0)
y_pred = model_1.predict(x_train_1)
'''
clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=150, objective="binary:logistic", subsample=0.5, colsample_bytree=0.5, seed=0, min_child_weight=10)
#cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
#score = cross_val_score(clf, x_train_1, y_train_1, scoring="neg_log_loss", cv = cv, n_jobs=-1, verbose=1)

t = time.time()
clf.fit(x_train_1, y_train_1)
print("Training time %lf"%(time.time()-t))

y_pred_train = clf.predict_proba(x_train_1)
y_pred_train = [i[1] for i in y_pred_train]

y_pred_test = clf.predict_proba(x_test)
y_pred_test = [i[1] for i in y_pred_test]


x_train_2, y_train_2, x_test, id_test, label_encoder_2 = load_preprocessed_data(train_path_2, test_path)

y_pred_train = np.array(y_pred_train)
y_pred_train = y_pred_train[:,None]

y_pred_test = np.array(y_pred_test)
y_pred_test = y_pred_test[:,None]
print(y_pred_train[:10])

x_train_2 = np.hstack((x_train_2, y_pred_train))
x_test = np.hstack((x_test, y_pred_test))

def initializeDefaultParams():
    params = {}
    params['objective'] = "multi:softprob"
    params['num_class'] = 12

    return params

def initializeDefaultParamsGPU():
    params = initializeDefaultParams()
    params['tree_method'] = 'gpu_hist'
    return params

num_boost_round_default = 200
seed = 0
metrics = ['mlogloss']

train_dmat = xgb.DMatrix(x_train_2, y_train_2)

params = initializeDefaultParamsGPU()

t = time.time()

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_mlogloss= float("Inf")
best_params = None
for subsample, colsample in gridsearch_params:
    print("CV with subsample={}, colsample={}".format(subsample,colsample))

    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    cv_results = xgb.cv(
        params,
        train_dmat,
        num_boost_round=num_boost_round_default,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=20
    )

    # Update best score
    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tmlogloss {} for {} rounds\n".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (subsample, colsample)

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))

subsample = best_params[0]
colsample = best_params[1]
params['subsample'] = subsample
params['colsample_bytree'] = colsample

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(2,10)
    for min_child_weight in range(2,10)
]

min_mlogloss = float("Inf")
best_params = None
#search for max depth and min_child_weight
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    cv_result = xgb.cv(
        params,
        train_dmat,
        num_boost_round=num_boost_round_default,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=20
    )

    mean_mlogloss = cv_result['test-mlogloss-mean'].min()
    boost_rounds = cv_result['test-mlogloss-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (max_depth,min_child_weight)

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))
max_depth = best_params[0]
min_child_weight = best_params[1]

params['max_depth'] = max_depth
params['min_child_weight'] = min_child_weight

#search eta
list_eta = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
min_mlogloss= float("Inf")
best_params = None

for eta in list_eta:
    print("CV with eta={}".format(eta))

    params['eta'] = eta

    cv_results = xgb.cv(
        params,
        train_dmat,
        num_boost_round=num_boost_round_default,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=20
    )

    # Update best score
    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tmlogloss {} for {} rounds\n".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = eta

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))

eta = best_params
params['eta'] = eta

n_estimators = [25, 50, 100, 150, 200, 250]
min_mlogloss= float("Inf")
best_params = None

#search n estimator
for n in n_estimators:
    print("n with eta={}".format(n))

    cv_results = xgb.cv(
        params,
        train_dmat,
        num_boost_round=n,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=20
    )

    # Update best score
    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tmlogloss {} for {} rounds\n".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = n 

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))
num_boost_round = best_params

print("Training time %lf"%(time.time()-t))
print("Overall best parameter: max_depth {}, min_child_weight {}, eta {}, num_boost_round {} subsample {} colsample {}".format(max_depth, min_child_weight, eta, num_boost_round, subsample, colsample))







