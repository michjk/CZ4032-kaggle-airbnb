import numpy as np 
import pandas as pd
import os
import time

from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

#dataset path
dataset_folder_path = os.path.dirname(__file__) + "dataset/"
train_path_1 = dataset_folder_path + "train_users_2_NDF_vs_non_NDF.csv"
train_path_2 = dataset_folder_path + "train_users_exclude_NDF.csv"
test_path = dataset_folder_path + "test_users.csv"
output_path = "submission.csv"
cv = 5

x_train_1, y_train_1, x_test, id_test, label_encoder_1 = load_preprocessed_data(train_path_1, test_path)

model_1 = RandomForestClassifier(n_estimators=500, n_jobs=2, min_samples_leaf=3)
model_1.fit(x_train_1, y_train_1)
prob_1 = model_1.predict_proba(x_test)

x_train_2, y_train_2, x_test, id_test, label_encoder_2 = load_preprocessed_data(train_path_2, test_path)

model_2 = RandomForestClassifier(n_estimators=500, n_jobs=2, min_samples_leaf=3)
model_2.fit(x_train_2, y_train_2)
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

print(np.sum(prob_3[0]))

create_kaggle_submission_csv(prob_3, id_test, label_encoder_3, output_path)
