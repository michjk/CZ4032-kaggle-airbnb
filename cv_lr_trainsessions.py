import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#dataset path
base_path = os.path.dirname(os.getcwd())
train_path = base_path + "/dataset/train_session_v8.csv"
test_path = base_path + "/dataset/test_session_v8.csv"
output_path = base_path + "/dataset/submission_cv_lr.csv"

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#train
print("Initiating LogisticRegression Classifier")
clf = LogisticRegression(penalty='l2', C=100, random_state=0, n_jobs=-1)
t = time.time()
print("Starting CV 5-fold")
scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy', verbose=10)
print("Accuracy scores:")
for score in scores:
    print(score)
print("Mean score:")
print(scores.mean())
print("Training time %lf"%(time.time()-t))
print("Training LogisticRegression Classifier")
clf.fit(x_train, y_train)
print("Predicting probabilities")
y_pred = clf.predict_proba(x_test)

print("Creating submission.csv")
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)