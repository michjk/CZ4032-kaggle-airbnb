import numpy as np 
import pandas as pd
import time
import os
from data_manager.data_preprocessor import load_preprocessed_data, load_label_encoder
from data_manager.data_generator import create_kaggle_submission_csv_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#dataset path
base_path = os.path.dirname(__file__)
dataset_folder_path = "dataset/"
train_path = base_path + dataset_folder_path + "train_users_2.csv"
train_ndf_path = base_path + dataset_folder_path + "train_users_2_NDF_vs_non_NDF.csv"
train_countries_path = base_path + dataset_folder_path + "train_users_exclude_NDF.csv"
test_path = base_path + dataset_folder_path + "test_users.csv"
output_path = base_path + "submission_randomforest_2.csv"
cv = 5

x_train_ndf, y_train_ndf, x_test_ndf, id_test, label_encoder_ndf = load_preprocessed_data(train_ndf_path, test_path)
x_train_countries, y_train_countries, x_test_countries, id_test, label_encoder_countries = load_preprocessed_data(train_countries_path, test_path)
label_encoder = load_label_encoder(train_path)

# ndf and non-ndf classifier
rf_ndf = RandomForestClassifier(n_estimators=100, n_jobs=2, min_samples_leaf=3)
t = time.time()
scores_ndf = cross_val_score(rf_ndf, x_train_ndf, y_train_ndf, cv = cv)
print("Mean CV score of training for ndf and non-ndf: %lf"%scores_ndf.mean())
print("Training time for ndf and non-ndf: %lf"%(time.time()-t))
rf_ndf.fit(x_train_ndf, y_train_ndf)

# countries classifier
rf_countries = RandomForestClassifier(n_estimators=100, n_jobs=2, min_samples_leaf=3)
t = time.time()
scores_countries = cross_val_score(rf_countries, x_train_countries, y_train_countries, cv = cv)
print("Mean CV score of training for countries: %lf"%scores_countries.mean())
print("Training time for countries: %lf"%(time.time()-t))
rf_countries.fit(x_train_countries, y_train_countries)

# prediction
y_pred_ndf = rf_ndf.predict_proba(x_test_ndf)
y_pred_countries = rf_countries.predict_proba(x_test_countries)

create_kaggle_submission_csv_2(y_pred_ndf, y_pred_countries, id_test, label_encoder, label_encoder_ndf, label_encoder_countries, output_path)