import numpy as np 
import pandas as pd
import os
from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.ensemble import RandomForestClassifier

#dataset path
base_path = os.path.dirname(os.getcwd())
train_path = base_path + "/dataset/train_session_v8.csv"
test_path = base_path + "/dataset/test_session_v8.csv"
output_path = base_path + "/dataset/submission_rf.csv"

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#train
print("Initiating RandomForestClassifier")
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, min_samples_leaf=10, max_features=20)
print("Training RandomForestClassifier")
clf.fit(x_train, y_train)
print("Predicting probabilities")
y_pred = clf.predict_proba(x_test)

print("Creating submission.csv")
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)
