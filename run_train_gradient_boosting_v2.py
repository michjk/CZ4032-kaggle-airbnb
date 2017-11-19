import numpy as np 
import xgboost as xgb 
import pandas as pd
import time
from data_manager.data_preprocessor import load_preprocessed_data
from data_manager.data_generator import create_kaggle_submission_csv
from sklearn.model_selection import cross_val_score

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_session.csv"
test_path = dataset_folder_path + "test_session.csv"
output_path = "submission.csv"
cv = 5

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)

#5 fold score
clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=150, objective="multi:softprob", min_child_weight=4, subsample=1.0, colsample_bytree=1.0, seed=0)
t = time.time()
scores = cross_val_score(clf, x_train, y_train, cv = cv, scoring="accuracy")
print(scores.mean())
print("Validation time %lf"%(time.time()-t))

#train
t = time.time()
clf.fit(x_train, y_train)
print("Training time %lf"%(time.time()-t))

#predict and create submission csv
y_pred = clf.predict_proba(x_test)
create_kaggle_submission_csv(y_pred, id_test, label_encoder, output_path)