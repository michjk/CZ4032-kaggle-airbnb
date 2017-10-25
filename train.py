import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import load_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#dataset path
base_path = os.path.dirname(__file__)
dataset_folder_path = "dataset/"
train_path = base_path + dataset_folder_path + "train_users_2.csv"
test_path = base_path + dataset_folder_path + "test_users.csv"
output_path = base_path + "submission_randomforest.csv"
cv = 5

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)

#train
clf = RandomForestClassifier(n_estimators=100, n_jobs=2, min_samples_leaf=3)
t = time.time()
scores = cross_val_score(clf, x_train, y_train, cv = cv)
print(scores.mean())
print("Training time %lf"%(time.time()-t))
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)

create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)