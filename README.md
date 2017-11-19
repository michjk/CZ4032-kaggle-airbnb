# CZ4042: Airbnb New User Bookings Competition

This repository contains code for CZ4042 course. The project is chosen from Kaggle. The competition can be found here https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

The program is run in python 2.7

To install dependency:
'''
pip install -r requirements.txt
'''

To run the python files:
'''
python file_name.py
'''

The code name for classifier is gradient_boosting, lr(logistic regression), rf(random forest)

There are 3 types of file:
1. Training file
The file is for training model. The prefix is run_train*.py. If training 2 layer, the prefix is run_train_stack*.py. If the dataset is train+session, the suffix is v2.
2. cross validation
This file is used to do parameter tuning. The prefix is run_model_selection*.py.
3. Feature engineering
There is a ipynb jupyter file for feature engineering. The file is session_preprocessing for analyzing and preprocessing session file.
To run jupyter server
'''
jupyter notebook
'''
After that, go to the provided link address and select the jupyter file.

The dataset provided:
1. train_users.csv
2. test_users.csv
3. train_session_v8.csv : train_users + session
4. test_session_v8.csv : test_users + session
5. train_users_2_NDF_vs_non_NDF.csv : train_users with label NDF and non NDF
6. train_users_exclude_NDF.csv : train_users with label non NDF
7. train_session_v8_ndf_vs_non_ndf.csv : train_users + session with label NDF and non NDF
8. train_session_v8_exclude_ndf.csv : train_users + session with label non NDF



