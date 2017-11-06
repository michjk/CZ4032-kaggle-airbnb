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

print("ok")
''' #insert result from classifier 1 to train 2
for i in range(len(x_train_2)):
    x_train_2[i].append(y_pred_train[i])

for i in range(len(x_test)):
    x_test[i].append(y_pred_test[i])
'''

clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, objective="multi:softprob", subsample=0.8, colsample_bytree=0.2, seed = 0)

t = time.time()
print(x_train_2[-1][-1])
print(y_pred_train[-1])
clf.fit(x_train_2, y_train_2)
print("Training time %lf"%(time.time()-t))

y_pred_test_2 = clf.predict_proba(x_test)

create_kaggle_submission_csv(y_pred_test_2, id_test, label_encoder_2, output_path)








