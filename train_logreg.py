import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import load_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#dataset path
base_path = os.path.dirname(__file__)
train_path = base_path + "dataset/train_users_2.csv"
test_path = base_path + "dataset/test_users.csv"
output_path = base_path + "dataset/submission.csv"
cv = 5

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
print("Finish data preprocessing")

#train
print("Initiating Logistic Regression Model")
clf = LogisticRegression(penalty='l2', C=100, random_state=0, n_jobs=-1)
t = time.time()
print("Starting CV")
scores = cross_val_score(clf, x_train, y_train, cv=cv, scoring='accuracy', verbose=10)
print("Accuracy scores:")
for score in scores:
    print(score)
print("Mean score:")
print(scores.mean())
print("Training time %lf"%(time.time()-t))
clf.fit(x_train, y_train)
print("Predicting probabilities")
y_pred = clf.predict_proba(x_test)

print("Creating submission.csv")
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)
