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
train_path_1 = dataset_folder_path + "train_session_v8_ndf_vs_non_ndf.csv"
train_path_2 = dataset_folder_path + "train_session_v8_exclude_ndf.csv"
test_path = dataset_folder_path + "test_users.csv"
output_path = "submission.csv"
cv = 5

x_train_1, y_train_1, x_test, id_test, label_encoder_1 = load_preprocessed_data(train_path_1, test_path)

model_1 = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, objective="binary:logistic", seed=0, min_child_weight=2, subsample=1.0, colsample_bytree=0.5)

t = time.time()
kfold = StratifiedKFold(n_splits = cv, shuffle=True, random_state=0)
val_score = cross_val_score(model_1, x_train_1, y_train_1, scoring='accuracy', cv=kfold)
print("Validation time %lf"%(time.time()-t))
print("Mean validation score: ", val_score.mean())

t = time.time()
model_1.fit(x_train_1, y_train_1)
print("Training time %lf"%(time.time()-t))
prob_1 = model_1.predict_proba(x_test)

x_train_2, y_train_2, x_test, id_test, label_encoder_2 = load_preprocessed_data(train_path_2, test_path)

model_2 = xgb.XGBClassifier(max_depth=3, learning_rate = 0.1, n_estimators=135, objective="multi:softprob", subsample=1.0, colsample_bytree=0.3, seed=0, min_child_weight=6)

t = time.time()
kfold = StratifiedKFold(n_splits = cv, shuffle=True, random_state=0)
val_score = cross_val_score(model_2, x_train_2, y_train_2, scoring='accuracy', cv=kfold)
print("Validation time %lf"%(time.time()-t))
print("Mean validation score: ", val_score.mean())

t = time.time()
model_2.fit(x_train_2, y_train_2)
print("Training time %lf"%(time.time()-t))
prob_2 = model_2.predict_proba(x_test)

label_encoder_3 = createHybridLabelEncoder(label_encoder_1, label_encoder_2)

prob_3 = []
len_test = len(x_test)
len_country = len(prob_2[0])
NDF_pos = np.where(label_encoder_3.classes_ == 'NDF')[0][0]

for i in range(len_test):
    tmp = []
    prob_2[i] = [prob_1[i][1]*num for num in prob_2[i]]

    for j in range(NDF_pos):
        tmp.append(prob_2[i][j])
    
    tmp.append(prob_1[i][0])
    
    for j in range(NDF_pos, len_country):
        tmp.append(prob_2[i][j])
    
    prob_3.append(tmp)

create_kaggle_submission_csv(prob_3, id_test, label_encoder_3, output_path)

