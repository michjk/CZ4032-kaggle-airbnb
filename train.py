import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import load_direct_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#dataset path
base_path = os.path.dirname(os.getcwd())
train_path = base_path + "/dataset/train_session_preprocessed.csv"
test_path = base_path + "/dataset/test_session_preprocessed.csv"
output_path = base_path + "/dataset/submission.csv"
cv = 5

x_train, y_train, x_test, id_test, label_encoder = load_direct_preprocessed_data(train_path, test_path)

#train
clf = RandomForestClassifier(n_estimators=500, n_jobs=2, min_samples_leaf=3)
t = time.time()
scores = cross_val_score(clf, x_train, y_train, cv = cv)
print(scores.mean())
print("Training time %lf"%(time.time()-t))
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)

create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)
