import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import *
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#dataset path
base_path = os.path.dirname(os.getcwd())
train_path = base_path + "/dataset/train_users_2.csv"
test_path = base_path + "/dataset/test_users.csv"
output_path = base_path + "/dataset/submission_cv_rf_trainonly.csv"

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#train
print("Initiating RandomForestClassifier")
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, min_samples_leaf=10, max_features=20)
t = time.time()
print("Starting CV 5-fold")
scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy', verbose=10)
print("Accuracy scores:")
for score in scores:
    print(score)
print("Mean score:")
print(scores.mean())
print("Training time %lf"%(time.time()-t))
print("Training RandomForestClassifier")
clf.fit(x_train, y_train)
print("Predicting probabilities")
y_pred = clf.predict_proba(x_test)

print("Creating submission.csv")
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)