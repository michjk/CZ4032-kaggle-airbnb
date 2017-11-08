import numpy as np 
import xgboost as xgb 
import pandas as pd
import time
from data_manager.data_preprocessor import load_preprocessed_data
from data_manager.data_preprocessor import load_direct_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_session_preprocessed.csv"
test_path = dataset_folder_path + "test_session_preprocessed.csv"
output_path = "submission.csv"
cv = 5

x_train, y_train, x_test, id_test, label_encoder = load_direct_preprocessed_data(train_path, test_path)

#cross validation
clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, objective="multi:softprob", subsample=0.5, colsample_bytree=0.5, seed=0)

kfold = StratifiedKFold(n_splits=cv, shuffle = True)
t = time.time()
scores = cross_val_score(clf, x_train, y_train, cv=kfold, verbose=5)
print("Cross validation time %lf"%(time.time()-t))
print("validation accuracy %.3f"%(scores.mean()))

#fit training data and prediction
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)

#write csv file for kaggle submission
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)