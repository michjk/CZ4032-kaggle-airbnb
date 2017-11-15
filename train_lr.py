import numpy as np 
import pandas as pd
import os
from data_manager.data_preprocessor import load_direct_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.linear_model import LogisticRegression

#dataset path
base_path = os.path.dirname(os.getcwd())
train_path = base_path + "/dataset/train_session_v8.csv"
test_path = base_path + "/dataset/test_session_v8.csv"
output_path = base_path + "/dataset/submission_lr.csv"

x_train, y_train, x_test, id_test, label_encoder = load_direct_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#train
print("Initiating Logistic Regression Classifier")
clf = LogisticRegression(penalty='l2', C=100, random_state=0, n_jobs=-1)
print("Training Logistic Regression Classifier")
clf.fit(x_train, y_train)
print("Predicting probabilities")
y_pred = clf.predict_proba(x_test)

print("Creating submission.csv")
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)
